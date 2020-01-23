#!/bin/sh
if [ "$1" = 'rsync_server' ]; then
  mkdir -p /root/.ssh
  sed -i "s/#PasswordAuthentication yes/PasswordAuthentication no/g" /etc/ssh/sshd_config
  sed -i 's/root:!/root:*/' /etc/shadow
  sed -i 's/#HostKey \/etc\/ssh\/ssh_host_rsa_key/HostKey \/etc\/ssh\/ssh_host_rsa_key/g' /etc/ssh/sshd_config


  SSH_PARAMS="-D -e -p ${SSH_PORT:-22} $SSH_PARAMS"
  echo "Running: /usr/sbin/sshd $SSH_PARAMS"
  exec /usr/sbin/sshd -D $SSH_PARAMS

  echo "Running: /usr/bin/rsync --no-detach --daemon"
  exec /usr/bin/rsync --no-detach --daemon
fi
exec "$@"
