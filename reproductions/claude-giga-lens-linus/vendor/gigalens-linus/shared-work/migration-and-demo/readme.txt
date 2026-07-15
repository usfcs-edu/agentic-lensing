
Hi! This shared-work folder will hopefully be the place to put a large chunk of our work, to enable better compatibility and easy crossover.
    It is sequestered in a folder specifically to enable you to easily filter folders if you want to have a branch with other files (which will almost certainly happen). 
    See a discussion on how to implement automatic filtering here: https://gist.github.com/wizioo/c89847c7894ede628071
    Otherwise, you can use the staged change manager via the extension or the 'git add' command to manually select the files you want to commit on each branch.

To migrate to the new kernel version, simply run the script in ./migration/, or read through it and do as you please. If you find some issue with the modern_env package list,
    please bring it up asap, as stabilizing dependency versions is a top priority. Currently cuda-12 packages are included alongside cuda-13 packages (and the cuda 13 provided by NERSC).
    You can check what version of cuda you are using, and activate cuda 13, with 'module avail cuda' and 'module load cuda13.0'.
    I can only promise IWMM (it works on my machine), so if you run into an error, then that is good data.

That aside, the demo is in an ipynb file here. It doesn't leverage every new feature that this branch has wrt master, but should work out of the box. It uses the files in ../../wip/
    as currently working versions. These files are functional in the new version for me, but I haven't tested them as extensively as I'm sure you have, so if possible use them as a 
    reference to update your version, and replace the /wip/ files with your corrected ones. You can use this demo as a test bench to ensure it still works, but also running your own
    code is going to be important to find and cover edge cases. For example: NUTS works for the demo data, but I haven't tested it for the multiplane case. 

If a piece of code seems to be working, stable, and final (e.g. the new lstsq simulator), consider moving it out of /wip/ and into a more final location inside /src/ once you've verified
    these conditions. If the file is intended to be outside of /src/, maybe we should consider creating a /extensions/ folder, or it can live in /shared-work/.
    Ideally, /wip/ is not a permanent home for files.

note: I haven't looked at the new shapelets code yet, it should work but it is untested

Anyways, glhf!