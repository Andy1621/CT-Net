#!/bin/bash
this_dir=`pwd`
export PYTHONPATH=$this_dir:$this_dir/ops:$this_dir/arch:$this_dir/cam:$PYTHONPATH
echo "Done! Now the PYTHONPATH is as follows"
echo $PYTHONPATH