import json
import boto3

# Endpoint name of deployed model
ENDPOINT = "sagemaker-scikit-learn-2021-12-16-07-08-18-576"
CLIENT = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):

    # Get event data
    body = event['body']
    data = json.loads(body)["data"]

    # Preparing output
    output = {}
    output["person"] = data["person"]
    output["offer"] = data["offer"]
    output["time_in_days_received"] = data["time_in_days_received"]

    # Make a prediction
    response = CLIENT.invoke_endpoint(
        EndpointName=ENDPOINT,
        Body=json.dumps(data),
        ContentType="application/json"
    )
    response_content = json.loads(response["Body"].read().decode('utf-8'))
    output["response_probability"] = response_content["response_probability"]

    return {
        'statusCode': 200,
        'body': json.dumps(output)
    }