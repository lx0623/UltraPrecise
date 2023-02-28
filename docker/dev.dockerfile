FROM rateup/aries:rateup-cuda10.1-base
RUN apt-get install -y openssh-server
RUN  mkdir -p /var/run/sshd 
RUN apt-get install -y net-tools vim
RUN touch /i_am_docker.txt
WORKDIR /rateup

EXPOSE 3306

RUN echo "/usr/local/cuda/compat/" >> /etc/ld.so.conf.d/cuda.conf
RUN ldconfig

RUN mkdir -p mkdir/root/.ssh/
RUN echo root:123456 | chpasswd
RUN rm /etc/ssh/ssh_host_rsa_key && ssh-keygen -q -t rsa -b 2048  -f /etc/ssh/ssh_host_rsa_key -P '' -N ''
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config 

EXPOSE 22

CMD  /usr/sbin/sshd -D
