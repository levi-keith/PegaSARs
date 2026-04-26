# Auto-parsed AWS SAR demo

This is the stripped-down Streamlit app.

Only user input: an existing S3 video URI.

The app parses the camera ID and recording start time from the S3 object filename, loads the matching ruleset from `camera_rules.json`, calls TwelveLabs Pegasus through Bedrock, evaluates the deterministic rule engine, and generates SARs only for violations.

Expected filename:

```text
camera_01__20260425_210000__north_entry_road.mp4
```

Run from the SageMaker Studio terminal where `aws sts get-caller-identity` works:

```bash
python -m streamlit run app.py --server.port 8505 --server.address 0.0.0.0
```

Open the SageMaker proxy for port 8505.
