"""CPU parallel script generates training data for the SHAMnet11 initializer
by calling the generate_traindata_batch function in generate_shamnet11_traindata.
"""
import h5py
import os
from mpi4py import MPI
from glob import glob
import argparse
import numpy as np
from subprocess import check_output, CalledProcessError
from time import time
from jax import random as jran
from shamnet.generate_shamnet_traindata import generate_traindata_batch


PREPAT = "SHAMnet_tdata_{0}"
RANKPAT = "rank_{0}_epoch_{1}.h5"
BEBOP_DRN = "/lcrc/project/halotools/random_data/0802"
TASSO_DRN = "/Users/aphearin/work/DATA/random_data/tmp"
FIXED_KEYLIST = list(
    (
        "fixed_smf_params",
        "fixed_shmf_params",
        "fixed_scatter_params",
        "fixed_smhm_params",
    )
)


def write_batch(X, Y, losses, fixed_p_dicts, outdrn, bpat, bounds_X, smhm_bounds):
    outname = os.path.join(outdrn, bpat)
    fixed_smf_p, fixed_shmf_p, fixed_scatter_p, fixed_smhm_p = fixed_p_dicts
    with h5py.File(outname, "w") as hdf:
        hdf["traindata_X"] = X
        hdf["traindata_Y"] = Y
        hdf["traindata_loss"] = losses
        hdf["X_mins"] = bounds_X[0]
        hdf["X_maxs"] = bounds_X[1]
        hdf["smhm_params_mins"] = smhm_bounds[0]
        hdf["smhm_params_maxs"] = smhm_bounds[1]
        for key, value in fixed_smf_p.items():
            hdf["fixed_smf_params/{}".format(key)] = value
        for key, value in fixed_shmf_p.items():
            hdf["fixed_shmf_params/{}".format(key)] = value
        for key, value in fixed_scatter_p.items():
            hdf["fixed_scatter_params/{}".format(key)] = value
        for key, value in fixed_smhm_p.items():
            hdf["fixed_smhm_params/{}".format(key)] = value


def cleanup(drn, nick):
    pat = "*" + nick + "*" + RANKPAT.format("*", "*")
    fnames = glob(os.path.join(drn, pat))
    X_collector = []
    Y_collector = []
    loss_collector = []
    for fn in fnames:
        with h5py.File(fn, "r") as hdf:
            X_collector.append(hdf["traindata_X"][...])
            Y_collector.append(hdf["traindata_Y"][...])
            loss_collector.append(hdf["traindata_loss"][...])
    all_traindata_X = np.concatenate(X_collector)
    all_traindata_Y = np.concatenate(Y_collector)
    all_traindata_losses = np.concatenate(loss_collector)
    outname = os.path.join(drn, job_nickname + ".h5")
    with h5py.File(outname, "w") as hdf:
        hdf["traindata_X"] = all_traindata_X
        hdf["traindata_Y"] = all_traindata_Y
        hdf["traindata_loss"] = all_traindata_losses

        with h5py.File(fnames[0], "r") as hdfin:
            hdf["X_mins"] = hdfin["X_mins"][...]
            hdf["X_maxs"] = hdfin["X_maxs"][...]
            hdf["smhm_params_mins"] = hdfin["smhm_params_mins"][...]
            hdf["smhm_params_maxs"] = hdfin["smhm_params_maxs"][...]
            for fixed_key in FIXED_KEYLIST:
                if fixed_key in hdfin.keys():
                    for subkey in hdfin[fixed_key].keys():
                        outkey = "{0}/{1}".format(fixed_key, subkey)
                        hdf[outkey] = hdfin[outkey][...]

    for fn in fnames:
        command = "rm " + fn
        try:
            check_output(command, shell=True)
        except CalledProcessError:
            pass

    return all_traindata_X.shape[0]


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    jkey = jran.PRNGKey(rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("job_nickname", help="Used in output fname")
    parser.add_argument("-n_batch", help="Batch size", type=int, default=10)
    parser.add_argument("-n_steps", help="Steps per batch", type=int, default=1000)
    parser.add_argument("-n_warmup", help="# warmup steps", type=int, default=2)
    parser.add_argument(
        "-outdir",
        help="Name of the output file directory",
        default="BEBOP",
        choices=("TASSO", "BEBOP"),
    )
    args = parser.parse_args()

    n_steps = args.n_steps
    n_warmup = args.n_warmup
    if args.outdir == "TASSO":
        outdrname = TASSO_DRN
    elif args.outdir == "BEBOP":
        outdrname = BEBOP_DRN
    else:
        raise ValueError("Bad output drname")
    job_nickname = PREPAT.format(args.job_nickname)

    start = time()

    n_per_rank = max(1, int(args.n_batch / nranks))
    max_per_epoch = 100
    n_per_epoch = min(n_per_rank, max_per_epoch)
    n_complete_epochs, leftover = divmod(n_per_rank, n_per_epoch)
    n_epochs = n_complete_epochs + bool(leftover)
    msg = "Computing n_per_rank = {0} n_per_epoch = {1} and n_epochs = {2}"
    if rank == 0:
        print(msg.format(n_per_rank, n_per_epoch, n_epochs))

    epoch = 0
    for i in range(n_epochs):
        oldkey, jkey = jran.split(jkey)
        _ret = generate_traindata_batch(n_per_epoch, n_steps, n_warmup, jkey)
        (
            traindata_X,
            traindata_Y,
            traindata_loss,
            fixed_params,
            bounds_X,
            smhm_bounds,
        ) = _ret

        outbase = "_".join((job_nickname, RANKPAT.format(rank, epoch)))
        write_batch(
            traindata_X,
            traindata_Y,
            traindata_loss,
            fixed_params,
            outdrname,
            outbase,
            bounds_X,
            smhm_bounds,
        )
        epoch += 1
    #
    comm.Barrier()
    end = time()
    if rank == 0:
        n_tot = cleanup(outdrname, job_nickname)
        msg = (
            "For {0} total points with {1} total ranks, wall-clock time {2:.1f} seconds"
        )
        print(msg.format(n_tot, nranks, end - start))
