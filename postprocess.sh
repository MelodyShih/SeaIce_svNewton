dt=$(grep "dt" $1 | sed 's/.*days//g')
echo $dt
if [ $3 == "stdnewton" ] 
then 
    grep "stdnewton" $1 > tmp.txt
    sed -i 's/,.*//g' tmp.txt
    sed -i 's/stdnewton: \#iter//g' tmp.txt
    if [ $# -eq 4 ]
    then
        grep "stressvel" $4 > tmp2.txt
        sed -i 's/,.*//g' tmp2.txt
        sed -i 's/stressvel: \#iter//g' tmp2.txt
    fi
else 
    grep "stressvel" $1 > tmp.txt
    sed -i 's/,.*//g' tmp.txt
    sed -i 's/stressvel: \#iter//g' tmp.txt
fi 

if [ $# -eq 4 ]
then
	python3 plotdata.py $dt day $2 1
else
	python3 plotdata.py $dt day $2 0
fi
