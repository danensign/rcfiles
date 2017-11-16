# Path settings

export PYTHONPATH=$HOME/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=$SPARK_HOME/libexec/python:$SPARK_HOME/libexec/python/lib/py4j-0.10.4-src.zip:$PYTHONPATH

# Alias 
alias awk=mawk
alias ls="ls --color"
alias python=python3
alias rm="rm -i"
alias top=htop
alias vi=vim

# Hadoop aliases
#hadoop_path=$(brew --prefix hadoop)
alias hstart="$hadoop_path/sbin/start-dfs.sh;$hadoop_path/sbin/start-yarn.sh"
alias hstop="$hadoop_path/sbin/stop-yarn.sh;$hadoop_path/sbin/stop-dfs.sh"

# User experience
source ~/.my_prompt
command -v fortune >/dev/null 2>&2 && fortune | cowsay -f stegosaurus 

if [ -f /usr/share/bash-completion/completions/git ]
then
    source /usr/share/bash-completion/completions/git
fi

