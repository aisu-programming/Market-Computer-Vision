# Footfall_Detection
if ! [ -s Footfall_Detection/footfall-detection.pt ]
then
    cd Footfall_Detection
    wget https://github.com/aisu-programming/Footfall-Detection/releases/download/v1.0/footfall-detection.pt
    cd ..
fi


# Vegetable_Instance_Segmentation
if ! [ -s Vegetable_Instance_Segmentation/vegetable-instance-segmentation.h5 ]
then
    cd Vegetable_Instance_Segmentation
    wget https://github.com/aisu-programming/Vegetable-Instance-Segmentation/releases/download/v1.0/vegetable-instance-segmentation.h5
    cd ..
fi


# Shelf_Empty_Detection
cd Shelf_Empty_Detection
if ! [ -s shelf-empty-detection-640-0.75.pt ]
then
    wget https://github.com/aisu-programming/Shelf-Empty-Detection/releases/download/v1.0/shelf-empty-detection-640-0.75.pt
fi
if ! [ -s shelf-empty-detection-256-1.0.pt ]
then
    wget https://github.com/aisu-programming/Shelf-Empty-Detection/releases/download/v1.1/shelf-empty-detection-256-1.0.pt
fi
if ! [ -s shelf-empty-detection-256-0.75.pt ]
then
    wget https://github.com/aisu-programming/Shelf-Empty-Detection/releases/download/v1.2/shelf-empty-detection-256-0.75.pt
fi
if ! [ -s machine01-shelf-empty-detection-256-0.75.pt ]
then
    wget https://github.com/aisu-programming/Shelf-Empty-Detection/releases/download/v1.3/machine01-shelf-empty-detection-256-0.75.pt
fi
if ! [ -s machine02-shelf-empty-detection-256-0.75.pt ]
then
    wget https://github.com/aisu-programming/Shelf-Empty-Detection/releases/download/v1.3/machine02-shelf-empty-detection-256-0.75.pt
fi
cd ..