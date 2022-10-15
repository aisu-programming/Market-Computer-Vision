if ! [ -s Footfall_Detection/footfall-detection.pt ]
then
    cd Footfall_Detection
    wget https://github.com/aisu-programming/Footfall-Detection/releases/download/v1.0/footfall-detection.pt
    cd ..
fi

if ! [ -s Vegetable_Instance_Segmentation/vegetable-instance-segmentation.h5 ]
then
    cd Vegetable_Instance_Segmentation
    wget https://github.com/aisu-programming/Vegetable-Instance-Segmentation/releases/download/v1.0/vegetable-instance-segmentation.h5
    cd ..
fi

cd Shelf_Empty_Detection
rm shelf-empty-detection-0.75.pt
wget https://github.com/aisu-programming/Shelf-Empty-Detection/releases/download/v1.0/shelf-empty-detection-0.75.pt
# rm shelf-empty-detection-1.0.pt
# wget https://github.com/aisu-programming/Shelf-Empty-Detection/releases/download/v1.0/shelf-empty-detection-1.0.pt
cd ..