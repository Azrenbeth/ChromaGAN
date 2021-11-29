import h5py
import json
import sys

BATCH_SIZE = 4

assert len(sys.argv) == 2

with h5py.File(sys.argv[1], "r+") as f:
	model_config = json.loads(f.attrs["model_config"])
	model_config["config"]["layers"][1]["config"]["stateful"] = True
	model_config["config"]["layers"][0]["config"]["batch_input_shape"][0] = BATCH_SIZE
	f.attrs["model_config"] = json.dumps(model_config)
