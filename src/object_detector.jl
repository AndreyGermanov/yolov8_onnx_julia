using Images, ONNXRunTime, Genie, Genie.Router, Genie.Requests, Genie.Renderer.Json

# Main function, starts the web server
function main()    
    # handle site root. Returns index.html file content
    route("/") do 
        String(read("index.html"))
    end 

    # handle /detect POST requests: receive an image from frontend
    # and returns array of detected objects in a format: [x1,y1,x2,y2,class,probability]
    # to a web browser as a JSON
    route("/detect", method=POST) do
        buf = IOBuffer(filespayload()["image_file"].data)
        json(detect_objects_on_image(buf))
    end

    up(8080, host="0.0.0.0", async=false)
end

# Function receives an uploaded image file body
# passes it through the YOLOv8 neural network
# model and returns an array of bounding boxes of 
# detected objects where each bounding box is an 
# array in a format [x1,y1,x2,y2,object_class,probability]
function detect_objects_on_image(buf)
    input, img_width, img_height = prepare_input(buf)
    output = run_model(input)
    return process_output(output, img_width, img_height)
end

# Function resizes image to a size,
# supported by default Yolov8 neural network (640x640)
# converts in to a tensor of (1,3,640,640) shape
# that supported as an input to a neural network
function prepare_input(buf)
    img = load(buf)
    img_height, img_width = size(img)
    img = imresize(img,(640,640))
    img = RGB.(img)
    input = channelview(img)
    input = reshape(input,1,3,640,640)
    return Float32.(input), img_width, img_height    
end

# Function receives an input image tensor,
# passes it through the YOLOv8 neural network
# model and returns the raw object detection
# result
function run_model(input)
    model = load_inference("yolov8m.onnx")
    outputs = model(Dict("images" => input))
    return outputs["output0"]
end

# Function receives the raw object detection
# result from neural network and converts
# it to an array of bounding boxes. Each
# bounding box is an array of the following format:
# [x1,y1,x2,y2,object_class,probability]
function process_output(output, img_width, img_height)
    output = output[1,:,:]
    output = transpose(output)

    boxes = []
    for row in eachrow(output)        
        prob = maximum(row[5:end])
        if prob<0.5
            continue
        end
        class_id = Int(argmax(row[5:end]))
        label = yolo_classes[class_id]
        xc,yc,w,h = row[1:4]
        x1 = (xc-w/2)/640*img_width
        y1 = (yc-h/2)/640*img_height
        x2 = (xc+w/2)/640*img_width
        y2 = (yc+h/2)/640*img_height
        push!(boxes,[x1,y1,x2,y2,label,prob])
    end

    boxes = sort(boxes, by = item -> item[6], rev=true)
    result = []
    while length(boxes)>0
        push!(result,boxes[1])
        boxes = filter(box -> iou(box,boxes[1])<0.7,boxes)
    end
    return result
end

# Calculates "Intersection-over-union" coefficient
# for specified two boxes
# https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
# Each box should be provided in a format:
# [x1,y1,x2,y2,object_class,probability]
function iou(box1,box2)
    return intersect(box1,box2) / union(box1,box2)
end

# Calculates union area of two boxes
# Each box should be provided in a format:
# [x1,y1,x2,y2,object_class,probability]
function union(box1,box2)
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[1:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[1:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersect(box1,box2)
end

# Calculates intersection area of two boxes
# Each box should be provided in a format:
# [x1,y1,x2,y2,object_class,probability]
function intersect(box1,box2)
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[1:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[1:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)
end

# Array of YOLOv8 class labels
yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

main()