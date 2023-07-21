from importlib_resources import files


ANGLE_LIST_DIR = files('pytom_tm.angle_lists')
AVAILABLE_ROTATIONAL_SAMPLING = {
    '7.00': ['angles_7.00_45123.txt', 45123],
    '35.76': ['angles_35.76_320.txt', 320],
    '19.95': ['angles_19.95_1944.txt', 1944],
    '90.00': ['angles_90.00_26.txt', 26],
    '18.00': ['angles_18.00_3040.txt', 3040],
    '12.85': ['angles_12.85_7112.txt', 7112],
    '38.53': ['angles_38.53_256.txt', 256],
    '11.00': ['angles_11.00_15192.txt', 15192],
    '17.86': ['angles_17.86_3040.txt', 3040],
    '25.25': ['angles_25.25_980.txt', 980],
    '50.00': ['angles_50.00_100.txt', 100],
    '3.00': ['angles_3.00_553680.txt', 553680],
}
for v in AVAILABLE_ROTATIONAL_SAMPLING.values():
    v[0] = ANGLE_LIST_DIR.joinpath(v[0])


def load_angle_list(file_name):
    with open(str(file_name)) as fstream:
        lines = fstream.readlines()
    return [tuple(map(float, x.strip().split(' '))) for x in lines]


def convert(angle, order_in='zxz', order_out='zyz'):
    pass
