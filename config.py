import scipy.io as sio

Is_AwA_dataset = False  # True for using AwA dataset, False for using Cub dataset
# import data and variables from the saved mat_file
if Is_AwA_dataset:
    system_config = {
        "M": 1,  # number of samples per test
        "tau": 0.008,  # latency constraint (default 0.008)
        "interval": 50,  # the max duration length for an actor with constant speed (default 50)
        "max_speed": 6,  # m/s (default 6)
    }
    mat_file = "./data/variables_awa.mat"  # AwA dataset
    trans_num_SKL = (
        5  # number of classes in the transmitter knowledge space (default 5)
    )
    receiv_num_SKL = 8  # number of classes in the receiver knowledge space (default 8)
    mec_num_SKL = (
        10  # number of classes in the MEC server knowledge space (default 10), new add
    )
else:
    system_config = {
        "M": 1,  # number of samples per test
        "tau": 0.003,  # latency constraint (default 0.003)
        "interval": 50,  # the max duration length for an actor with constant speed (default 50)
        "max_speed": 6,  # m/s (default 6)
    }
    mat_file = "./data/variables_cub.mat"  # Cub dataset
    trans_num_SKL = (
        25  # number of classes in the transmitter knowledge space (default 25)
    )
    receiv_num_SKL = (
        40  # number of classes in the receiver knowledge space (default 40)
    )
    mec_num_SKL = (
        50  # number of classes in the MEC server knowledge space (default 50), new add
    )

mat_contents = sio.loadmat(mat_file)

Visual_trans = mat_contents["Visual_trans"]
Semantic_trans = mat_contents["Semantic_trans"]
kopt_trans = mat_contents["kopt_trans"]
Visual_rec = mat_contents["Visual_rec"]
Semantic_rec = mat_contents["Semantic_rec"]
Semantic_mec = mat_contents["Semantic_trans"]  # new add for mec server
X_te = mat_contents["X_te"]
num_te_id = mat_contents["num_te_id"]
te_cl_id = mat_contents["te_cl_id"]
S_te_pro = mat_contents["S_te_pro"]

P_U = 10 ** (10 / 10.0) * 10 ** (-3)
beta0 = 10 ** (-30 / 10.0)
N0 = 4 * 10 ** (-21) * 10**6
B = 10**5
