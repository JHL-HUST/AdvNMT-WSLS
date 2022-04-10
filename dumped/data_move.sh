exp_type=gogr
res_type=gogr_res
f_dir=en_de_en/transformer
buf_dir=TRANSFORMERoracle/

for((i=0;i<1;i++));
do
    s_dir=$f_dir/$exp_type/job$i/$buf_dir/data
    mkdir ../$res_type/job$i
    mkdir ../$res_type/job$i/data
    t_dir=../$res_type/job$i/data
    cp $s_dir/* $t_dir/
done