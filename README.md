# Learn SSD
1. Keras Layers
## Anchor box
- Phần tinh túy nhất của SSD có lẽ là việc xác định các layers output của anchor box hoạc default bounding box ở các feature map.
- Anchor box layer sẽ nhận đầu vào là một feature map có kích thước( f_w, f_h, n_channels) và các scales, aspect,
ratios, trả ra đầu ra là một tensor kích thước(f_w, f_h, n_boxes,4) trong đó chiều cuối cùng đại diện cho 4 offsets của bounding box
như mô tả trong default box và tỷ lệ cạnh( aspect ratio)
## Code biến đổi khá phức tạp:
- bước 1: từ scale, size( giá trị lớn nhất của width và height), và aspect ratio ta xác định kích thước các cạnh của các boudnig box theo công thức:
        BOX_H = SCALE * SIZE / sqrt(ASPECT RATIO)
        BOX_W = SCALE * SIZE * sqrt(ASPECT RATIO)
- bước 2: từ các cell trên feature map chiếu lại trên ảnh input image để thu được step khoảng cách giữa các center point của mỗi cell theo công thức:
        STEP_H = IMG_H / FEATURE_MAP_H
        STEP_W = IMG_W / FEATURE_MAP_W
- bước 3: tính tọa độ các điểm (cx,cy,w,h) trên hình ảnh gốc dựa trên phép linear interpolation qua hàm np.linspace():
        Cx = np.linspace(SRART_W, END_W, FEATURE_MAP_W)
        Cy = np.linspace(SRART_H, END_H, FEATURE_MAP_H)
## kết quả trả về là một tensor có shape là (f_w, f_h, n_boxes, 8), trong đó chiều cuối cùng = 8 tương ứng với 4 offsets của default bounding box và 4 variances đại diện cho các scales của default bounding box.

2. Các bước thực hiện để khởi tạo cấu trúc của mạng ssd_300 bao gồm:

- Bước 1: Xây dựng kiến trúc mạng bao gồm:
- Bước 1.1: Xây dựng kiến trúc mạng base network theo VGG16 đã loại bỏ các fully connected layers ở cuối.
- Bước 1.2: Áp dụng các convolutional filter có kích thước (3 x 3) để tính toán ra features map.
- Bước 1.3: Xác định output phân phối xác suất theo các classes ứng với mỗi một default bounding box.
- Bước 1.4: Xác định output các tham số offset của default bounding boxes tương ứng với mỗi cell trên các features map.
- Bước 1.5: Bước 1.5: Tính toán các AnchorBoxes làm cơ sở để dự báo offsets cho các predicted bounding boxes bao quan vật thể. Gía trị của các AnchorBoxes chỉ hỗ trợ trong quá trình tính toán offsets và không xuất hiện ở output như giá trị cần dự báo.
- Bước 2: Reshape lại các output để đưa chúng về kích thước của (feature_map_w, feature_map_h, n_boxes, -1). Trong đó -1 đại diện cho chiều cuối cùng được tính dựa vào các chiều còn lại theo hàm reshape.
- Bước 3: Liên kết các khối tensorflow output của bước 2 được tính từ confidence, các offsets của bounding box và các offsets của anchor box.
- Bước 4: Kết nối với output. Thêm layers softmax trước confidence của bounding box.
