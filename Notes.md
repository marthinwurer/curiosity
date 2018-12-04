#Installation stuff:

##ViZDoom:

custom install from repo, install dependencies from ZDoom:

https://zdoom.org/wiki/Compile_ZDoom_on_Linux

Then install the ones from https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux


Vizdoom cmake flags:

https://stackoverflow.com/a/38121972/3000741

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON3=ON \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
```

make

manually copy the build into pyenv

```bash
cp bin/python3.7/pip_package/ ~/.pyenv/versions/pytorch/lib/python3.7/site-packages/vizdoom -R
```

Tutorial here:

http://vizdoom.cs.put.edu.pl/tutorial


## Q learning

https://youtu.be/w33Lplx49_A?t=3387

Q(s,a) <- (1 - alpha)Q(s,a) + alpha * reward

from the deepmind paper:

The training target y for the given action a that transitions from s to s' with a reward r and discount d is equal to:

r if a final state

r + d * argmaxQ(s', a)

We want Q(s, a) to equal y. 

Therefore, the loss is y - Q(s, a). 

This gets squared for mse loss, but google clipped it to -1, 1 before squaring for stability reasons.


## Malmo

https://github.com/Microsoft/malmo/blob/master/doc/build_linux.md





