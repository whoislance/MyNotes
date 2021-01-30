### 1. 在store下载wsl2

​	windows下设置：

​	`控制面板--程序与功能--启动或关闭Windows功能`勾选`适用于Linux的Windows子系统`，`重启`即可

### 2. 修改镜像源

```bash
cd /etc/apt/
sudo mv sources.list sources.list.old
sudo vim sources.list
#输入镜像链接[下方]
sudo apt-get update
```

阿里镜像源：https://developer.aliyun.com/mirror/ubuntu?spm=a2c6h.13651102.0.0.3e221b11EJqkSP

```
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
```

### 3. VSCode中打开

```
code .
```

### 4. [python] 安装pyenv

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
source ~/.bashrc

sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev
```

安装特定版本python：

```bash
• pyenv versions #列出目前系統中所有安裝的 Python
• pyenv version #顯示目前預設的 Python 版本
• pyenv global <python_version> #設定預設 Python 版本
• pyenv install <python_version> #安裝特定版本的 Python
• pyenv uninstall <python_version> #移除特定版本的 Python

e.g.1: pyenv install 3.6.8
e.g.2: v=3.6.8;wget https://npm.taobao.org/mirrors/python/$v/Python-$v.tar.xz -P ~/.pyenv/cache/;pyenv install $v 
```

安装package

```
numpy==1.17.5
matplotlib==3.1.3
pandas==0.25.3
scipy==1.4.1
torch==1.2.0
torchvision==0.4.0
tensorflow==1.15.0
keras==2.2.5
pillow==6.2.2
tqdm==4.28.1
opencv-python==4.1.2.30
gensim==3.6.0
scikit-learn==0.22.1
scikit-image==0.16.2
lime==0.1.1.37
nltk==3.4.5
h5py==2.10.0
pyyaml==5.3.1
shap==0.35.0
gym[box2d]==0.17.0
```

设置镜像

```shell
pip install pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
pip install -i http://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
or
pip --default-timeout=1000 install -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com 
```



### 5. [c++]

