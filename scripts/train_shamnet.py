"""
"""
import argparse
from time import time
import os
import subprocess
from jax import random as jran
from jax.experimental import optimizers as jax_opt

from shamnet.generate_shamnet_traindata import shamnet_loss_data_generator
from shamnet.sham_network import get_sham_network
from shamnet.load_shamnet_traindata import load_initializer_tdata_and_vdata
from shamnet.jaxnn_helpers import data_stream
from shamnet.jaxnn_helpers import train_with_single_steps_per_batch
from shamnet.jaxnn_helpers import write_trained_network_to_h5
from shamnet.generate_shamnet_traindata import X_MINS, X_MAXS
from shamnet.sham_network import LOGSM_PRED_MIN, LOGSM_PRED_MAX

SEED = 3
SHAMNET_DESIGN = (64, 32, 8, 4)
TASSO_TDATA_DRN = "/Users/aphearin/work/DATA/SHAMnet_data"
BEBOP_TDATA_DRN = "/lcrc/project/halotools/SHAMnet_data"
TDATA_BASENAME = "SHAMnet_tdata_published_tdata.h5"
_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
NETDATA = X_MINS, X_MAXS, LOGSM_PRED_MIN, LOGSM_PRED_MAX, "Selu"
CLIP = -15.0

if __name__ == "__main__":
    start_script = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("outpat", help="Name of the output file pattern")
    parser.add_argument("n_steps", help="Number of steps per restart", type=int)
    parser.add_argument("n_restarts", help="Number of restarts", type=int)

    parser.add_argument(
        "-layers", help="Net architecture", nargs="*", type=int, default=SHAMNET_DESIGN
    )
    parser.add_argument("-n_steps0", help="Number initial steps", type=int, default=500)
    parser.add_argument(
        "-n_restarts0", help="Number of restarts for init", type=int, default=3
    )
    parser.add_argument("-outdrn", help="Output directory", default=_THIS_DRNAME)
    parser.add_argument("-indir", help="Tdata location", default="BEBOP")
    parser.add_argument(
        "-n_tdata", help="Size of training data", type=int, default=5_000
    )
    parser.add_argument("-timing_test", help="Time training", default=False, type=bool)
    parser.add_argument("-smf_batch_size", help="SMF batch size", default=50, type=int)
    parser.add_argument("-smf_lo", help="SMF lower bound", default=8.8, type=float)
    parser.add_argument("-smf_hi", help="SMF upper bound", default=12.2, type=float)
    parser.add_argument(
        "-smf_step_size", help="SMF step size", default=0.001, type=float
    )

    args = parser.parse_args()
    outpat = args.outpat
    n_steps = args.n_steps
    n_restarts = args.n_restarts
    n_steps_init = args.n_steps0
    outdrn = args.outdrn
    dense_layer_sizes = args.layers
    n_tdata = args.n_tdata
    n_tdata = args.n_tdata
    n_restarts0 = args.n_restarts0
    smf_batch_size = args.smf_batch_size
    smf_lo = args.smf_lo
    smf_hi = args.smf_hi
    smf_step_size = args.smf_step_size

    if args.timing_test:
        n_restarts0, n_restarts = 0, 0

    if args.indir == "TASSO":
        indir = TASSO_TDATA_DRN
    elif args.indir == "BEBOP":
        indir = BEBOP_TDATA_DRN
    else:
        raise ValueError("Bad input drname")

    s = "{}".format(dense_layer_sizes)
    network_string = "_".join(s[1:-1].replace(" ", "").split(","))
    outbase = outpat + "_" + network_string + ".h5"
    outname = os.path.join(outdrn, outbase)

    ran_key = jran.PRNGKey(0)

    _res = get_sham_network(dense_layer_sizes)

    net_init, net_pred, smhm_batch_loss = _res[:3]
    smf_batch_loss, smf_grads_batch_loss, smf_and_grads_joint_loss = _res[3:6]
    pred_logsm_batch, predict_smf_batch, predict_lnsmf_grads_batch = _res[6:]

    ran_key, p_key = jran.split(ran_key)
    init_net_params = net_init(ran_key=p_key)

    traindata_fn = os.path.join(indir, TDATA_BASENAME)

    train_key, ran_key = jran.split(ran_key)
    tdata, vdata = load_initializer_tdata_and_vdata(
        traindata_fn, n_tdata, train_key, 11, 15.5, n_mh=200, tol=-3.0, n_validate=200
    )
    X_tdata, Y_tdata = tdata[:2]

    batch_size_init = 50
    smhm_loss_data_generator = data_stream(batch_size_init, X_tdata, Y_tdata)
    loss_data = next(smhm_loss_data_generator)
    loss_init = smhm_batch_loss(init_net_params, loss_data)

    opt_init, opt_update, get_params = jax_opt.adam(0.001)
    opt_state = opt_init(init_net_params)

    if args.timing_test:
        _args = (
            n_steps_init,
            smhm_loss_data_generator,
            smhm_batch_loss,
            get_params,
            opt_state,
            opt_update,
        )
        start_initial_burn = time()
        _res = train_with_single_steps_per_batch(*_args)
        end_initial_burn = time()
        msg = "\nNumber of steps for initializer loss training = {0}"
        print(msg.format(n_steps_init))
        initial_burn_runtime = end_initial_burn - start_initial_burn
        msg = "Runtime for each iteration of the initial burn = {0:.2f} seconds\n"
        print(msg.format(initial_burn_runtime))

    # Train the initializer
    p_best = get_params(opt_state)
    fnames_to_cleanup = []
    loss_history_initializer = []
    for i in range(n_restarts0):
        outname_i = outname.replace(".h5", "initializer.{}.h5".format(i))
        start_tstep = time()
        opt_state = opt_init(p_best)
        ran_key, batch_size_key = jran.split(ran_key)
        batch_size = int(jran.randint(ran_key, minval=20, maxval=50, shape=(1,))[0])
        print("...Training iteration {0} with batch_size = {1}".format(i, batch_size))
        smhm_loss_data_generator = data_stream(batch_size, X_tdata, Y_tdata)
        _args = (
            n_steps_init,
            smhm_loss_data_generator,
            smhm_batch_loss,
            get_params,
            opt_state,
            opt_update,
        )
        _res = train_with_single_steps_per_batch(*_args)
        opt_state, gd_step, loss_history_i, p_best, loss_min = _res
        end_tstep = time()
        print("{0:.1f} seconds spent on this iteration".format(end_tstep - start_tstep))

        loss_history_initializer.extend(loss_history_i)
        _args = (outname_i, p_best, *NETDATA, loss_history_initializer)
        write_trained_network_to_h5(*_args)
        fnames_to_cleanup.append(outname_i)

    if args.timing_test is False:
        print("Writing initializer to disk\n")
        outname_init = outname.replace(".h5", ".initializer.h5")
        _args = (outname_init, p_best, *NETDATA, loss_history_initializer)
        write_trained_network_to_h5(*_args)
        for fname in fnames_to_cleanup:
            command = "rm -rf {}".format(fname)
            raw_result = subprocess.check_output(command, shell=True)

    # SMF-based loss training
    opt_init, opt_update, get_params = jax_opt.adam(smf_step_size)
    opt_state = opt_init(p_best)

    if args.timing_test:
        smf_loss_data_gen = shamnet_loss_data_generator(
            train_key, 75, 100, smf_lo, smf_hi, CLIP
        )
        _args = (
            n_steps,
            smf_loss_data_gen,
            smf_batch_loss,
            get_params,
            opt_state,
            opt_update,
        )
        start_smf_train = time()
        _res = train_with_single_steps_per_batch(*_args)
        end_smf_train = time()
        smf_train_runtime = end_smf_train - start_smf_train
        print("Number of steps for SMF-based loss training = {0}".format(n_steps))
        msg = "Runtime for each iteration of SMF-based loss training = {0:.2f} seconds"
        print(msg.format(smf_train_runtime))

    smf_loss_history = []
    smf_fnames_to_cleanup = []
    for i in range(n_restarts):
        outname_i = outname.replace(".h5", ".{}.h5".format(i))
        start_tstep = time()
        opt_state = opt_init(p_best)
        ran_key, batch_size_key, n_smf_bins_key, train_key = jran.split(ran_key, 4)
        smf_batch_size = int(
            jran.randint(batch_size_key, minval=20, maxval=100, shape=(1,))[0]
        )
        n_smf_bins = int(
            jran.randint(n_smf_bins_key, minval=50, maxval=150, shape=(1,))[0]
        )
        print(
            "...Training iteration {0} with batch_size = {1}".format(i, smf_batch_size)
        )
        smf_loss_data_gen = shamnet_loss_data_generator(
            train_key, smf_batch_size, n_smf_bins, smf_lo, smf_hi, CLIP
        )
        _args = (
            n_steps,
            smf_loss_data_gen,
            smf_batch_loss,
            get_params,
            opt_state,
            opt_update,
        )
        _res = train_with_single_steps_per_batch(*_args)
        opt_state, gd_step, loss_history_smf_i, p_best, loss_min = _res
        smf_loss_history.extend(loss_history_smf_i)
        end_tstep = time()
        msg = "{0:.1f} seconds spent on SMF loss training for iteration {1}\n"
        print(msg.format(end_tstep - start_tstep, i))

        _args = (outname_i, p_best, *NETDATA, smf_loss_history)
        write_trained_network_to_h5(*_args)
        smf_fnames_to_cleanup.append(outname_i)

    if args.timing_test is False:
        print("Writing trained SHAMNet to disk\n")
        outname = outname.replace(".h5", ".h5")
        _args = (outname, p_best, *NETDATA, smf_loss_history)
        write_trained_network_to_h5(*_args)
        for fname in smf_fnames_to_cleanup:
            command = "rm -rf {}".format(fname)
            raw_result = subprocess.check_output(command, shell=True)
