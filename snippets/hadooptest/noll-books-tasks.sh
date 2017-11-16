hadoop \
        jar /usr/local/opt/hadoop/libexec/share/hadoop/tools/lib/hadoop-streaming-2.8.1.jar \
        -D mapred.reduce.tasks=16
        -file /Users/densig249/code/hadooptest/mapper.py \
        -mapper /Users/densig249/code/hadooptest/mapper.py \
        -file /Users/densig249/code/hadooptest/reducer.py \
        -reducer /Users/densig249/code/hadooptest/reducer.py \
        -input /user/hduser/* \
        -output /user/hduser/output
