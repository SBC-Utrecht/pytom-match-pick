import threading
import cupy as cp
import voltools as vt
from cupyx.scipy.fftpack import get_fft_plan, fftn, ifftn
from tqdm import tqdm


class TemplateMatchingPlan():
    def __init__(self, volume, template, mask, wedge, get_fft_plan, deviceid):
        # Search volume + and fft transform plan for the volume
        self.volume = cp.asarray(volume, dtype=cp.float32, order='C')
        self.volume_ft = cp.fft.fftn(self.volume)
        self.fft_plan = get_fft_plan(self.volume.astype(cp.complex64))

        # Data for the mask
        self.mask = cp.asarray(mask, dtype=cp.float32, order='C')
        self.mask_texture = vt.StaticVolume(self.mask, interpolation='linear', device=f'gpu:{deviceid}')
        self.mask_padded = cp.zeros_like(self.volume).astype(cp.float32)
        self.mask_weight = self.mask.sum()  # weight of the mask

        # Init template data
        self.template = cp.asarray(template, dtype=cp.float32, order='C')
        self.template_texture = vt.StaticVolume(self.template, interpolation='filt_bspline', device=f'gpu:{deviceid}')
        self.template_padded = cp.zeros_like(self.volume)
        self.wedge = cp.asarray(wedge, order='C', dtype=cp.float32)  # fourier binary wedge weight for the template

        # Initialize result volumes
        self.ccc_map = cp.zeros_like(self.volume)
        self.scores = cp.ones_like(self.volume)*-1000
        self.angles = cp.ones_like(self.volume)*-1000

        # TODO needed?
        cp.cuda.stream.get_current_stream().synchronize()


class TemplateMatchingGPU(threading.Thread):
    def __init__(self, jobid, deviceid, volume, template, mask, wedge, angle_list, mask_is_spherical):
        threading.Thread.__init__(self)
        cp.cuda.Device(deviceid).use()

        self.Device = cp.cuda.Device
        self.jobid = jobid
        self.deviceid = deviceid
        self.active = True
        self.completed = False
        self.mask_is_spherical = mask_is_spherical  # whether mask is spherical
        self.angle_list = angle_list

        self.update_results = cp.ElementwiseKernel(
            'float32 scores, float32 angle_lists, float32 ccc_map, float32 angle_id',
            'float32 out, float32 out2',
            'if (scores < ccc_map) {out = ccc_map; out2 = angle_id;}',
            'update_results')

        self.plan = TemplateMatchingPlan(volume, template, mask, wedge, deviceid)

        print("Initialized job_{:03d} on device {:d}".format(self.jobid, self.deviceid))

    def run(self):
        print("Starting job_{:03d} on device {:d}".format(self.jobid, self.deviceid))
        self.Device(self.deviceid).use()
        self.template_matching_gpu()
        self.completed = True
        self.active = False

    def template_matching_gpu(self):
        # Size x template (sxz) and center x template (cxt)
        sxt, syt, szt = self.plan.template.shape
        cxt, cyt, czt = sxt // 2, syt // 2, szt // 2
        mx, my, mz = sxt % 2, syt % 2, szt % 2  # odd or even

        # Size x volume (sxv) and center x volume (xcv)
        sxv, syv, szv = self.plan.template_padded.shape
        cxv, cyv, czv = sxv // 2, syv // 2, szv // 2

        # Rotation center needs to be set as pytom default s // 2 + s % 2
        rotation_center = (cxt + mx, cyt + my, czt + mz)

        if self.mask_is_spherical:  # Then we only need to calculate std volume once
            self.plan.mask_padded[cxv - cxt:cxv + cxt + mx,
                                  cyv - cyt:cyv + cyt + my,
                                  czv - czt:czv + czt + mz] = self.plan.mask
            std_v = self.std_under_mask_convolution(self.plan.volume, self.plan.mask_padded, self.plan.mask_weight)

        # Track iterations with a tqdm progress bar
        for angle_id in tqdm(range(len(self.angle_list))):

            angles = self.angle_list[angle_id]

            if not self.mask_is_spherical:
                self.plan.mask_texture.transform(rotation=(angles[0], angles[2], angles[1]), rotation_order='rzxz',
                                                 output=self.plan.mask, center=rotation_center)
                self.plan.mask_padded[cxv - cxt:cxv + cxt + mx,
                                      cyv - cyt:cyv + cyt + my,
                                      czv - czt:czv + czt + mz] = self.plan.mask
                # Std volume needs to be recalculated for every rotation of the mask, expensive step
                std_v = self.std_under_mask_convolution(self.plan.volume, self.plan.mask_padded, self.plan.mask_weight)

            # Rotate template
            self.plan.template_texture.transform(rotation=(angles[0], angles[2], angles[1]), rotation_order='rzxz',
                                                 output=self.plan.template, center=rotation_center)

            # Add wedge to the template after rotating
            self.plan.template = cp.fft.irfftn(cp.fft.rfftn(self.plan.template) * self.plan.wedge,
                                               s=self.plan.template.shape).real

            # Normalize and mask template
            mean = self.mean_under_mask(self.plan.template, self.plan.mask, mask_weight=self.plan.mask_weight)
            std = self.std_under_mask(self.plan.template, self.plan.mask, mean, mask_weight=self.plan.mask_weight)
            self.plan.template = ((self.plan.template - mean) / std) * self.plan.mask

            # Paste in center
            self.plan.template_padded[cxv - cxt:cxv + cxt + mx,
                                      cyv - cyt:cyv + cyt + my,
                                      czv - czt:czv + czt + mz] = self.plan.template

            # Cross-correlate and normalize by std_v
            self.plan.ccc_map = self.normalized_cross_correlation(self.plan.volume_ft, self.plan.template_padded,
                                                                  std_v, self.plan.mask_weight, fft_plan=self.plan.fft_plan)

            # Update the scores and angle_lists
            self.update_results(self.plan.scores, self.plan.angles, self.plan.ccc_map,
                                angle_id, self.plan.scores, self.plan.angles)

    def is_alive(self):
        """
        whether process is running
        """
        return self.active

    def std_under_mask_convolution(self, volume, padded_mask, mask_weight):
        """
        std convolution of volume and mask
        """
        std_v = (
            self.mean_under_mask_convolution(volume ** 2, padded_mask, mask_weight) -
            self.mean_under_mask_convolution(volume, padded_mask, mask_weight) ** 2
        )
        std_v[std_v <= cp.float32(1e-09)] = 1
        return cp.sqrt(std_v)

    def mean_under_mask_convolution(self, volume, mask, mask_weight):
        """
        mean convolution of volume and mask
        """
        return (cp.fft.irfftn(cp.fft.rfftn(volume) * cp.fft.rfftn(mask).conj(), s=volume.shape) / mask_weight).real

    def mean_under_mask(self, volume, mask, mask_weight):
        """
        mean value of the template under the mask
        """
        return (volume * mask).sum() / mask_weight

    def std_under_mask(self, volume, mask, mean, mask_weight):
        """
        standard deviation of the template under the mask
        """
        return cp.sqrt(self.mean_under_mask(volume**2, mask, mask_weight) - mean**2)

    def normalized_cross_correlation(self, volume_fft, template, norm, mask_weight, fft_plan=None):
        """
        fast local correlation function between volume and template, norm is the standard deviation at each point in
        the volume in the masked area
        """
        return ifftn(volume_fft * fftn(template, plan=fft_plan).conj(), plan=fft_plan).real / (mask_weight * norm)

