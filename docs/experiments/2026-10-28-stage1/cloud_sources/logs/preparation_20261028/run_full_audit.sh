source /root/autodl-tmp/biyivla/logs/preparation_20261028/preparation_env.sh
cd /root/autodl-tmp/biyivla/openpi
"$BIYI_PYTHON" /root/autodl-tmp/biyivla/logs/preparation_20261028/biyivla_full_audit.py > /root/autodl-tmp/biyivla/logs/preparation_20261028/full_audit.log 2>&1
printf '%s\n' "$?" > /root/autodl-tmp/biyivla/logs/preparation_20261028/full_audit.exit
