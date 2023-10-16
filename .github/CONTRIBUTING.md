# Contributing

Any contribution is more the welcomed. If you are not sure about your contribution just push it in the in-process folder and I will see what can i do with it.

> **Tip:** If you are working from VS Code, you can [install the recommended extensions](https://dev.to/askrishnapravin/recommend-vs-code-extensions-to-your-future-teammates-4gkb) to quickly get setup.

## Working Locally

After forking and cloning your repo, You'll need to install a few dependencies

```sh
sudo apt install libeigen3-dev libgtest-dev libgmock-dev
```

Or you can use your favorite system installed.
If you prefer C++ package managers, you can do

- `vcpkg install eigen3 gtest`
- `conan install --requires="eigen/[>=3 <4]" --requires="gtest/[>=1 <2]" --generator=CMakeDeps --output-folder=build`

### Building the library

```sh
cmake -E make_directory build
cd build

cmake ..
cmake --build .
ctest .
```

If you have any questions feel free to reach out on discord.
