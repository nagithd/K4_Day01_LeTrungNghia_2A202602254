# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
- ```json
    {
      "sample_id": "traffic",
      "coco_image_id": 210273,
      "image_width": 640,
      "image_height": 428,
      "task": "image_classification",
      "taxonomy_name": "ImageNet-1K",
      "model_file": "yolo11n-cls.pt",
      "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
      "ultralytics_version": "8.4.145",
      "rank": 1,
      "class_id": 468,
      "class_name": "cab",
      "score": 0.510915
    }
  ```
- Record này phân loại ảnh vào lớp "cab"
- class list mà mô hình dự đoán do người dán nhãn quyết định.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? `class_id`để định danh ổn định cho máy xử lí, thông kê và liên kết với label map, trong `class_name`dễ dàng hơn để người dùng kiểm tra, báo cáo, taxonomy để xác định bộ dữ liệu nào đang được sử dụng.
- Với ảnh nhiều chủ thể, guideline cần quy định rõ “nhãn nào là nhãn chính”, vì bài toán `image_classification` thường chỉ trả một lớp cho cả ảnh. Nên quy định:
  - Có dùng single-label hay multi-label classification.
  - Tiêu chí chọn chủ thể chính: lớn nhất, gần camera nhất, nằm giữa ảnh, quan trọng nhất theo ngữ cảnh, hoặc theo annotation gốc.
  - Nếu nhiều chủ thể cùng mức độ quan trọng: gán nhiều nhãn hay loại ảnh đó khỏi tập single-label.
  - Có tính chủ thể bị che khuất, cắt mép ảnh hoặc quá nhỏ không.
  - Có gán nhãn background/uncertain/other khi không có chủ thể rõ ràng không.
  - Có giới hạn tối đa bao nhiêu chủ thể hoặc nhãn trên một ảnh không.
  - Cách đánh giá dự đoán: top-1, top-k, hoặc mAP/F1 cho multi-label.
- Model score là mô hình đang đánh giá kết quả dự đoán còn ground truth là bộ dữ liệu được người dùng quy chuẩn để đánh giá mô hình.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
- ```json
    {
      "sample_id": "traffic",
      "coco_image_id": 210273,
      "image_width": 640,
      "image_height": 428,
      "task": "object_detection",
      "taxonomy_name": "COCO-80",
      "model_file": "yolo11n.pt",
      "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
      "ultralytics_version": "8.4.145",
      "score_threshold": 0.35,
      "class_id": 5,
      "class_name": "bus",
      "score": 0.912558,
      "coordinate_unit": "pixel",
      "bbox_format": "xyxy",
      "bbox_xyxy": [
        93.17,
        187.95,
        223.01,
        320.91
      ],
      "bbox_width": 129.84,
      "bbox_height": 132.96
    }
  ```
- Diễn giải vị trí box bằng lời: bounding box bắt đầu từ điểm có tọa độ (93.17, 187.95) đến điểm cuối (223.01, 320.91) với chiều rộng 129.84px và chiều cao 132.96px
- Với object detection, số prediction là số box sau khi lọc confidence và NMS. Đối với threshold thấp: số lượng prediction sẽ nhiều hơn do giữ lại cả các prediction có độ tin cậy thấp, còn đối với threshold cao hơn ít prediction hơn nhưng độ tin cậy cao hơn.
- Khi giảm threshold, độ bao phủ tăng: hệ thống giữ lại nhiều vật thể hơn, đặc biệt các vật thể nhỏ, che khuất hoặc mờ. Đổi lại số false positive tăng, nên reviewer phải xem nhiều box hơn và tốn thời gian hơn. Khi tăng threshold, reviewer xem ít box hơn và danh sách “sạch” hơn, nhưng dễ bỏ sót vật thể thật có score thấp. Vì vậy coverage giảm.
- Đề xuất một quy tắc box chặt: Chỉ gửi cho reviewer các bounding box có `score >= 0.60`, sau khi áp dụng NMS với `IoU = 0.70`. Một box được xem là hợp lệ nếu diện tích tối thiểu 32×32 pixel, nằm trong biên ảnh, và không bị một box cùng lớp có score cao hơn chồng lấp từ 70% IoU trở lên. Các box có `0.35 <= score < 0.60` được gắn cờ cần kiểm tra.
- Đối với các vật thể bị che khuất hoặc cắt mép, tùy vào diện tích phần bị cắt có thể chọn giữa việc bỏ ảnh/vật thể hoặc dán nhãn phần nhìn thấy.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- ```json
  {
      "sample_id": "traffic",
      "coco_image_id": 210273,
      "image_width": 640,
      "image_height": 428,
      "task": "instance_segmentation",
      "taxonomy_name": "COCO-80",
      "model_file": "yolo11n-seg.pt",
      "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
      "ultralytics_version": "8.4.145",
      "score_threshold": 0.35,
      "instance_id": "traffic-003",
      "class_id": 2,
      "class_name": "car",
      "score": 0.890111,
      "coordinate_unit": "pixel",
      "bbox_format": "xyxy",
      "bbox_xyxy": [
        179.33,
        299.37,
        254.35,
        366.56
      ],
      "polygon_point_count": 59,
      "polygon_xy": [
        [
          201.0,
          300.0
        ],
        [
          200.0,
          301.0
        ],
        [
          199.0,
          301.0
        ],
        [
          197.0,
          303.0
        ],
        [
          196.0,
          303.0
        ],
        [
          193.0,
          306.0
        ],
        [
          193.0,
          307.0
        ],
        [
          191.0,
          309.0
        ],
        [
          191.0,
          310.0
        ],
        [
          189.0,
          312.0
        ],
        [
          189.0,
          313.0
        ],
        [
          186.0,
          316.0
        ],
        [
          186.0,
          317.0
        ],
        [
          184.0,
          319.0
        ],
        [
          184.0,
          320.0
        ],
        [
          183.0,
          321.0
        ],
        [
          183.0,
          322.0
        ],
        [
          182.0,
          323.0
        ],
        [
          182.0,
          334.0
        ],
        [
          181.0,
          335.0
        ],
        [
          181.0,
          339.0
        ],
        [
          180.0,
          340.0
        ],
        [
          180.0,
          362.0
        ],
        [
          181.0,
          363.0
        ],
        [
          181.0,
          365.0
        ],
        [
          182.0,
          366.0
        ],
        [
          215.0,
          366.0
        ],
        [
          216.0,
          365.0
        ],
        [
          233.0,
          365.0
        ],
        [
          234.0,
          366.0
        ],
        [
          246.0,
          366.0
        ],
        [
          249.0,
          363.0
        ],
        [
          249.0,
          361.0
        ],
        [
          250.0,
          360.0
        ],
        [
          250.0,
          349.0
        ],
        [
          251.0,
          348.0
        ],
        [
          251.0,
          327.0
        ],
        [
          254.0,
          324.0
        ],
        [
          254.0,
          321.0
        ],
        [
          252.0,
          319.0
        ],
        [
          251.0,
          319.0
        ],
        [
          250.0,
          318.0
        ],
        [
          249.0,
          318.0
        ],
        [
          247.0,
          316.0
        ],
        [
          247.0,
          315.0
        ],
        [
          245.0,
          313.0
        ],
        [
          245.0,
          312.0
        ],
        [
          244.0,
          311.0
        ],
        [
          244.0,
          310.0
        ],
        [
          243.0,
          309.0
        ],
        [
          243.0,
          308.0
        ],
        [
          242.0,
          307.0
        ],
        [
          242.0,
          306.0
        ],
        [
          241.0,
          305.0
        ],
        [
          241.0,
          304.0
        ],
        [
          240.0,
          303.0
        ],
        [
          240.0,
          302.0
        ],
        [
          239.0,
          301.0
        ],
        [
          239.0,
          300.0
        ]
      ]
    }
  ```
- Polygon ôm sát vật thể hơn so với bounding box do sử dụng nhiều điểm hơn
- `instance_id` là mã định danh duy nhất cho một vật thể cụ thể trong ảnh
- Đề xuất một quy tắc biên mask: Mask phải bao phủ toàn bộ phần vật thể nhìn thấy được, bám theo biên ảnh thực tế của đối tượng và không bao gồm nền. Mask không được vượt ra ngoài ảnh. Điểm polygon nằm trong khoảng `0 ≤ x < image_width` và `0 ≤ y < image_height`. Bbox phải là hộp nhỏ nhất bao kín mask.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  * Mờ: theo biên nhìn thấy đáng tin cậy; không “đoán” phần mờ vào nền. Nếu không phân biệt được biên, đánh dấu `ambiguous_boundary`.
  * Hai vật thể tiếp xúc: tạo hai `instance_id` và hai mask riêng; biên tách dựa trên đường ranh nhìn thấy. Nếu không có ranh rõ, escalation quyết định có gộp mask hay loại ảnh.
  * Che khuất: chỉ mask phần nhìn thấy, không suy diễn phần bị che phía sau vật khác.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ              | Đơn vị/định dạng ground truth                                                                | Lỗi hoặc điểm mơ hồ quan sát được                                                                                                          | Annotator làm gì?                                                                                                              | Reviewer xem gì?                                                                                                                                                                 |
| --------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh      | Một`class_id` theo taxonomy                                                                     | Nhiều chủ thể; không rõ chủ thể chính; nhãn ngoài taxonomy; ảnh mờ; ảnh không có đối tượng; lớp tương tự nhau.                | Áp dụng quy tắc chọn chủ thể chính hoặc gán đa nhãn                                                                   | Đúng taxonomy và class ID; chủ thể chính được chọn nhất quán; ảnh mơ hồ được xử lý đúng; không nhầm nhãn gần nhau.                                      |
| Phát hiện vật thể | Mỗi đối tượng là một annotation gồm`class_id`, `bbox_xyxy`                             | Box quá rộng/hẹp; trùng box; vật thể nhỏ; che khuất; bị cắt ở mép ảnh; nhiều vật thể chồng lấp; nhầm lớp.                        | Vẽ một box chặt cho từng vật thể nhìn thấy; chỉ box phần nằm trong ảnh; tạo instance riêng cho từng đối tượng | Box ôm sát vật thể; tọa độ nằm trong ảnh; không bỏ sót/nhân đôi đối tượng; class đúng; quy tắc che khuất và vật thể nhỏ được áp dụng nhất quán. |
| Instance segmentation | Mỗi thực thể gồm`instance_id`, `class_id`, polygon/RLE mask, `bbox_xyxy` suy ra từ mask | Mask lấn nền; hở phần vật thể; hai vật thể dính nhau; polygon tự cắt; biên mờ; vật thể bị che; mask quá nhỏ hoặc không hợp lệ. | Tạo một mask riêng cho mỗi instance; chỉ mask phần nhìn thấy; bám biên vật thể; không suy diễn phần che khuất    | Mask khớp biên; không chồng nền; instance chạm nhau được tách; polygon hợp lệ; bbox bao kín mask; cờ che khuất/cắt mép/mơ hồ chính xác.                      |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
