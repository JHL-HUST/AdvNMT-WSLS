exp_type=gogr
res_type=gogr_res
f_dir=dumped/en_de_en/transformer
buf_dir=TRANSFORMERoracle/
i_max = 39

for((i=0;i<$i_max;i++));
do
    s_dir=$f_dir/$exp_type/job$i/$buf_dir/data
    mkdir $f_dir/$res_type
    mkdir $f_dir/$res_type/job$i
    mkdir $f_dir/$res_type/job$i/data
    t_dir=$f_dir/$res_type/job$i/data
    cp $s_dir/* $t_dir/
done