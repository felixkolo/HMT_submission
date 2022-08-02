# python voronoi.py
# abaqus cae noGUI=main.py
# python image.py
echo "Bash version ${BASH_VERSION}..."
a=1
b=2000
for (( i=$a; i<=$b; i++ ))
do
    echo ""
    echo "#############################"
    echo "Simulation #$i of $b starting"
    echo "#############################"
    echo ""
    echo ""
    python voronoi.py
    abaqus cae noGUI=main.py
    python image.py
    rm -r ../Sim/SimFolders/$i/*
    rm -r ../Sim/Training/pic/*
    echo ""
    echo ""
    echo "#############################"
    echo "Simulation #$i of $b finished"
    echo "#############################"
    echo ""
    echo ""
done
rm -r ../Sim/SimFolders/*
echo ""
echo "#############################"
echo "All done, ciao ciao."
echo "#############################"

