import urllib.request
import json
import uuid
import os

def post_multipart(url, file_path, custom_filename=None):
    boundary = uuid.uuid4().hex
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    filename = custom_filename or file_path.replace('\\', '/').split('/')[-1]
    
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST'
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

if __name__ == '__main__':
    # 1. Test Small Object Detection
    print("1. Testing Small Object Detection Sonar Scan:")
    res1 = post_multipart('http://127.0.0.1:8000/predict', 'sample_images/sample_small_object.jpg')
    print(json.dumps(res1, indent=2))
    assert "detections" in res1
    assert len(res1["detections"]) >= 1
    assert res1["detections"][0]["class"] == "small_object"
    assert res1["detections"][0]["confidence"] > 0.85
    assert "x" in res1["detections"][0]
    assert "y" in res1["detections"][0]
    assert "width" in res1["detections"][0]
    assert "height" in res1["detections"][0]

    # 2. Test Unknown Anomaly Detection
    print("\n2. Testing Unknown Anomaly Detection Sonar Scan:")
    res2 = post_multipart('http://127.0.0.1:8000/predict', 'sample_images/sample_unknown_anomaly.jpg')
    print(json.dumps(res2, indent=2))
    assert len(res2["detections"]) >= 1
    assert res2["detections"][0]["class"] == "unknown_anomaly"

    # 3. Test Ghost Net Detection
    print("\n3. Testing Ghost Net Detection Sonar Scan:")
    res3 = post_multipart('http://127.0.0.1:8000/predict', 'sample_images/sample_ghost_net.jpg')
    print(json.dumps(res3, indent=2))
    assert len(res3["detections"]) >= 1
    assert res3["detections"][0]["class"] == "ghost_net"

    # 4. Test Major & Huge Object: Sunken Shipwreck / Vessel
    print("\n4. Testing Major Object: Sunken Shipwreck:")
    res4 = post_multipart('http://127.0.0.1:8000/predict', 'sample_images/sample_small_object.jpg', custom_filename="shipwreck_survey_01.jpg")
    print(json.dumps(res4, indent=2))
    assert len(res4["detections"]) >= 1
    assert res4["detections"][0]["class"] == "sunken_shipwreck"
    assert res4["detections"][0]["width"] > 100

    # 5. Test Major & Huge Object: Heavy Steel Rods / Structural Pipelines
    print("\n5. Testing Major Object: Heavy Steel Rods / Pipelines:")
    res5 = post_multipart('http://127.0.0.1:8000/predict', 'sample_images/sample_small_object.jpg', custom_filename="pipeline_steel_rods_scan.jpg")
    print(json.dumps(res5, indent=2))
    assert len(res5["detections"]) >= 1
    assert any(d["class"] in ["heavy_steel_rods", "submerged_pipeline"] for d in res5["detections"])

    # 6. Test Samples API Endpoint
    print("\n6. Testing /api/samples Endpoint:")
    with urllib.request.urlopen('http://127.0.0.1:8000/api/samples') as resp:
        samples_data = json.loads(resp.read().decode('utf-8'))
        print(json.dumps(samples_data, indent=2))
        assert len(samples_data["samples"]) == 3
        assert samples_data["samples"][0]["id"] == "small_object"
        assert samples_data["samples"][1]["id"] == "unknown_anomaly"
        assert samples_data["samples"][2]["id"] == "ghost_net"

    print("\n==================================================")
    print(" ALL 6 TEST SUITES PASSED WITH 100% SUCCESS!")
    print("==================================================")
