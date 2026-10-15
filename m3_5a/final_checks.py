import sys
from pathlib import Path
from .monitor import run
out=Path(sys.argv[1]).resolve()
print('final ML checks',flush=True)
run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','m3_5a/test_pipeline.py'],out/'monitor','final_tests',180)
run([sys.executable,'-m','m3_5a.verify_checkpoints',str(out)],out/'monitor','final_checkpoint_inference',180)
