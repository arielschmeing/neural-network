# 🤖 Neural Network - SampleRobot

> A Neural Network in C++ for the SampleRobot

---

## 📋 Prerequisites

Before you begin, make sure you have the following files available:

| File | Platform |
|------|----------|
| `mobilesim_0.9.8+ubuntu16_amd64.deb` | Linux |
| `libaria_2.9.4+ubuntu16_amd64.deb` | Linux |
| `msys2-x86_64-20250830.exe` | Windows |
| `MobileSim-0.7.5.exe` | Windows |
| `ARIA-2.9.1-1-x64.exe` | Windows |

---

## 🐧 Linux

### 1. Install dependencies

```bash
# MobileSim
sudo apt install ./mobilesim_0.9.8+ubuntu16_amd64.deb

# ARIA
sudo dpkg -i libaria_2.9.4+ubuntu16_amd64.deb

# G++ compiler
sudo apt update && sudo apt install g++
```

> 💡 Also install the **C/C++ extension** (by Microsoft) in VS Code.

### 2. Build

> ⚠️ It's recommended to run `make clean` before every build to avoid stale artifacts.

```bash
make clean
make
chmod +x ./build/main
```

### 3. Run

```bash
./build/main
```

---

## 🪟 Windows

### 1. Install in this exact order

1. `msys2-x86_64-20250830.exe`
2. `MobileSim-0.7.5.exe`
3. `ARIA-2.9.1-1-x64.exe`

### 2. Set up the compiler (MSYS2 terminal)

```bash
# Update and install GCC
pacman -Syu
pacman -S mingw-w64-ucrt-x86_64-gcc

# Verify the installation
export PATH="/c/msys64/ucrt64/bin:$PATH"
g++ --version
```

### 3. Add GCC to PATH permanently

```bash
nano ~/.bashrc
```

Add the following line at the end of the file:

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"
```

Then reload the configuration:

```bash
source ~/.bashrc
```

---

## 🚀 Quick Reference

| Action | Command |
|--------|---------|
| Clean build | `make clean` |
| Build | `make` |
| Run | `./build/main` |