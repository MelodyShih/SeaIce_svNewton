dt=$(grep "dt" $1 | sed 's/.*days//g')
echo $dt
grep "stdnewton:" $1 > tmp.txt
sed -i 's/,.*//g' tmp.txt
sed -i 's/stdnewton: \#iter//g' tmp.txt

python plotdata.py $dt day $2
