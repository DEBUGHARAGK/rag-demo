import json
import boto3
import urllib3
import hashlib
import os
import re
import time
import botocore.exceptions
from botocore.exceptions import BotoCoreError

# Initialize clients and resources
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_agent = boto3.client("bedrock-agent-runtime")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("AIState")
slackUrl = "https://slack.com/api/chat.postMessage"
SlackChatHistoryUrl = "https://slack.com/api/conversations.replies"
slackToken = os.environ.get("token")
http = urllib3.PoolManager()
model_arn = "arn:aws:bedrock:us-east-1:851725293109:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
KB_ID = "IB3AYZAYWQ"


# --- Knowledge Base Functions ---
def retrieve_from_kb(query, max_results=3):
    try:
        response = bedrock_agent.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": max_results}
            },
        )
        return response
    except (BotoCoreError, Exception) as e:
        print(f"Knowledge Base error: {str(e)}")
        return None


def process_kb_results(response):
    if not response or "retrievalResults" not in response:
        return ""
    return "\n\n".join(
        f"Source {i+1}: {result['content']['text']}"
        for i, result in enumerate(response["retrievalResults"])
        if result.get("content", {}).get("text")
    )


# --- Claude Invocation ---
def call_bedrock(messages):
    system_prompt = []
    conversation_history = []

    for msg in messages:
        if msg.startswith("System:"):
            system_prompt.append(msg.replace("System:", "").strip())
        else:
            conversation_history.append(msg)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.3,
        "system": "\n".join(system_prompt),
        "messages": [{"role": "user", "content": "\n".join(conversation_history)}],
    }
    body = json.dumps(payload)

    retries = 0
    max_retries = 5
    backoff = 1
    while retries < max_retries:
        try:
            response = bedrock.invoke_model(
                body=body,
                modelId=model_arn,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            if "content" in response_body and isinstance(
                response_body["content"], list
            ):
                completion = response_body["content"][0].get("text", "")
                return re.sub(r"^\s*Assistant: ?", "", completion).strip()
            return ""
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                time.sleep(backoff)
                retries += 1
                backoff *= 2
            else:
                raise
    raise Exception("Max retries exceeded")


# --- Enhanced Conversation Tracking ---
def hash_message(message):
    return hashlib.sha1(message.encode("utf-8")).hexdigest()


def get_conversation_state(user_id, thread_ts):
    try:
        response = table.get_item(
            Key={"id": user_id, "thread_ts": thread_ts or "direct"}
        )
        return response.get("Item", {})
    except Exception as e:
        print(f"DynamoDB get error: {str(e)}")
        return {}


def update_conversation_state(user_id, thread_ts, user_hash, bot_hash):
    try:
        table.put_item(
            Item={
                "id": user_id,
                "thread_ts": thread_ts or "direct",
                "last_user_hash": user_hash,
                "last_bot_hash": bot_hash,
                "timestamp": int(time.time()),
            }
        )
    except Exception as e:
        print(f"DynamoDB update error: {str(e)}")


# --- Lambda Handler ---
def lambda_handler(event, context):
    headers = {
        "Authorization": f"Bearer {slackToken}",
        "Content-Type": "application/json",
    }
    slackBody = json.loads(event["body"])
    slackEvent = slackBody.get("event", {})

    slackText = slackEvent.get("text", "")
    slackUser = slackEvent.get("user", "")
    channel = slackEvent.get("channel", "")
    thread_ts = slackEvent.get("thread_ts")
    ts = slackEvent.get("ts", "")
    eventType = slackEvent.get("type", "")
    subtype = slackEvent.get("subtype")
    bot_id = slackEvent.get("bot_id")

    current_user_hash = hash_message(slackText)
    state = get_conversation_state(slackUser, thread_ts)

    # Prevent duplicate processing
    if state.get("last_user_hash") == current_user_hash:
        if state.get("last_bot_hash"):
            return {"statusCode": 200, "body": "Already responded"}
        return {"statusCode": 200, "body": "Duplicate message"}

    # Process threaded replies
    if eventType == "message" and not bot_id and not subtype and thread_ts:
        kb_response = retrieve_from_kb(slackText)
        bedrockMsg = []

        if kb_context := process_kb_results(kb_response):
            bedrockMsg.append(f"System: Context:\n{kb_context}\n")

        historyResp = http.request(
            "GET",
            f"{SlackChatHistoryUrl}?channel={channel}&ts={thread_ts}",
            headers=headers,
        )
        history_data = json.loads(historyResp.data.decode("utf-8"))
        messages = history_data.get("messages", [])

        for message in messages:
            text = re.sub(r"<@.*?>", "", message.get("text", ""))
            if message.get("bot_profile"):
                bedrockMsg.append(f"Assistant: {text}")
            else:
                bedrockMsg.append(f"Human: {text}")

        bedrockMsg.append("Assistant:")
        msg = call_bedrock(bedrockMsg)

        http.request(
            "POST",
            slackUrl,
            headers=headers,
            body=json.dumps(
                {
                    "channel": channel,
                    "text": f"<@{slackUser}> {msg}",
                    "thread_ts": thread_ts,
                }
            ),
        )
        update_conversation_state(
            slackUser, thread_ts, current_user_hash, hash_message(msg)
        )

    # Process app mentions
    elif eventType == "app_mention" and not bot_id and not thread_ts:
        clean_msg = re.sub(r"<@.*?>", "", slackText)
        bedrockMsg = []

        kb_response = retrieve_from_kb(clean_msg)
        if kb_context := process_kb_results(kb_response):
            bedrockMsg.append(f"System: Context:\n{kb_context}\n")

        bedrockMsg.extend([f"Human: {clean_msg}", "Assistant:"])
        msg = call_bedrock(bedrockMsg)

        http.request(
            "POST",
            slackUrl,
            headers=headers,
            body=json.dumps(
                {"channel": channel, "text": f"<@{slackUser}> {msg}", "thread_ts": ts}
            ),
        )
        update_conversation_state(slackUser, None, current_user_hash, hash_message(msg))

    return {"statusCode": 200, "body": json.dumps({"msg": "processed"})}
