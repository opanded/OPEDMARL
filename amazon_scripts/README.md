
# 使用Amazon EC2上的MPI的教程

这是一个基于MPI和Amazon EC2构建分布式计算系统的教程。

## Amazon EC2计算实例的配置说明 
这里有一些安装MPI和mpi4py包以及配置ssh服务以实现实例之间通信的说明。您也可以直接使用Amazon EC2上的公共AMI镜像，它已经准备好使用了。

### 一个可直接使用的公共Amazon EC2 AMI 
请确保选择区域US East (Ohio) us-east-2。这个AMI是匿名的，可以直接使用。

```
AMI ID:  ami-05a7b2deb8470f517

owner:  753214619702
```

### 配置Amazon EC2实例的步骤

- Install MPI
```
sudo apt install openmpi-bin openmpi-dev openmpi-common openmpi-doc libopenmpi-dev
```

- Install mpi4py
```
pip3 install mpi4py
```

- 配置ssh以实现实例之间无密码登录
请参考这个链接作为参考：https://www.tecmint.com/ssh-passwordless-login-using-ssh-keygen-in-5-easy-steps/

- 配置ssh以实现无需手动输入’yes’的登录:
请参考这个链接作为参考：https://unix.stackexchange.com/questions/33271/how-to-avoid-ssh-asking-permission

在您配置好实例后，您可以提交配置并制作一个AMI供以后使用。

### 在创建实例时编辑安全组
编辑安全组以允许您的实例的所有流量，这样ssh才能工作。



## 用于协调分布式计算系统的Shell脚本

### 创建实例并提取IP地址信息
在Amazon EC2网站上创建实例来构建一个分布式计算系统，它由一个主节点和多个工作节点组成。将Amazon实例信息复制到文件`amazon_instances_info`中，例如：
```
–	i-0f7f74cf0420a75a3	c5n.large	52.15.165.68	2021/08/08 09:11 GMT-7
–	i-003c00df228882bfd	c5n.large	18.218.185.168	2021/08/08 09:11 GMT-7
–	i-062f6166ac671a077	c5n.large	52.15.88.70	2021/08/08 09:11 GMT-7
```
这里总共有三个实例，让第一个实例作为主节点，其余两个实例作为工作节点。我们需要IP地址信息来进行通信。在上面的例子中，52.15.165.68是主节点的IP地址。我们运行以下脚本来获取所有节点的IP地址，并将其存储到文件`nodeIPaddress`中
```
./get_ip_address.sh
```
或者
```
awk '{ print $4 }' amazon_instances_info > nodeIPaddress
```
它提取了文件`amazon_instances_info`中第四列的字符串，并将它们存储在文件`nodeIPaddress`中。


### 登录主节点
运行脚本登录主节点
```
./login_master.sh 1
```
命令后面的索引表示第i个节点。在这种情况下，'1'表示第一个节点或主节点。`login_master.sh`的内容如下：
```
#!/usr/bin/ksh
ARRAY=()
while read LINE
do
    ARRAY+=("$LINE")
done < nodeIPaddress
ssh  -i ~/AmazonEC2/.ssh/linux_key_pari.pem ubuntu@${ARRAY[$1]}
```
其中`~/AmazonEC2/.ssh/linux_key_pari.pem`是在Amazon EC2上创建实例时生成和下载的密钥对权限。在这种情况下，我的密钥对的名字是`linux_key_pari.pem`，放在目录`~/AmazonEC2/.ssh/`下。您需要根据您的情况相应地修改它。您可能还需要使用`apt-get install ksh`来安装`ksh`库来运行脚本。

### 在主节点上运行MPI程序
下一步是在主节点上运行MPI程序，例如，`train_darl1n.py`。在运行MPI程序之前，您需要将`train_darl1n.py`放在所有节点的同一目录下。（您可以使用Shell脚本`transferFile.sh`从本地主机上传文件到Amazon EC2，这将在后面解释。）然后您可以运行脚本
```
./run_spread.sh
```
或其他类似的脚本。文件`nodeIPaddress`应该与主节点中的`run_spread.sh`在同一目录下。

### 辅助脚本

- `./transfer.sh`: 从本地主机传输文件到文件`nodeIPaddress`中列出的节点，例如，
`./transfer.sh /home/smile/aaai_darl1n/setup.py /home/ubuntu/aaai_darl1n/`将本地主机上的文件`/home/smile/aaai_darl1n/setup.py`传输到Amazon EC2实例目录`/home/ubuntu/aaai_darl1n/`中。


- `./download_file.sh`: 从Amazon EC2实例下载文件到本地主机。

- `./ExecuteCommandAllNodes.sh`: 在所有节点上执行一个命令
```
i=1
filename='nodeIPaddress'
while IFS= read -r line
do
echo $line
echo "Execution Done!"
ssh -i  ~/AmazonEC2/.ssh/linux_key_pari.pem -n ubuntu@$line 'killall python3'
((i=i+1))
done < $filename
```
它在所有节点上执行命令`killall python3`。
