import { SendMessageCommand, SQSClient } from "@aws-sdk/client-sqs";
import { getWebSecret } from "@/lib/secrets";
import { safeEqual } from "@/lib/security";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Matches the FIFO message group used by the reconciliation schedule, so the
// worker applies every change to the activity data one at a time.
const EVENTS_MESSAGE_GROUP = "strava-events";

const sqs = new SQSClient({});

// Strava retries any non-2xx response and counts failures against the
// subscription, so events this deployment will never process are acknowledged
// and dropped. Only a transient enqueue failure asks Strava to retry.
const acknowledge = () => new Response(null, { status: 200 });

export async function GET(request: Request) {
  const url = new URL(request.url);
  const mode = url.searchParams.get("hub.mode");
  const token = url.searchParams.get("hub.verify_token");
  const challenge = url.searchParams.get("hub.challenge");

  let expected: string;
  try {
    expected = (await getWebSecret()).webhook_verify_token;
  } catch (error) {
    console.error("[webhook] verify token unavailable", error);
    return NextResponse.json({ error: "Webhook is not configured" }, { status: 503 });
  }
  if (mode !== "subscribe" || !token || !safeEqual(token, expected)) {
    return NextResponse.json({ error: "Webhook verification failed" }, { status: 403 });
  }
  if (!challenge) {
    return NextResponse.json({ error: "Missing challenge" }, { status: 400 });
  }
  return NextResponse.json({ "hub.challenge": challenge });
}

export async function POST(request: Request) {
  let event: Record<string, unknown>;
  try {
    event = (await request.json()) as Record<string, unknown>;
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  // Athlete events (deauthorization) need no processing here.
  if (event.object_type === "athlete") return acknowledge();
  if (
    event.object_type !== "activity" ||
    !["create", "update", "delete"].includes(String(event.aspect_type)) ||
    typeof event.object_id !== "number" ||
    typeof event.owner_id !== "number" ||
    typeof event.subscription_id !== "number"
  ) {
    return NextResponse.json({ error: "Invalid activity event" }, { status: 400 });
  }

  const configuredSubscription = process.env.STRAVA_SUBSCRIPTION_ID;
  const configuredAthlete = process.env.STRAVA_ATHLETE_ID;
  const queueUrl = process.env.STRAVA_EVENTS_QUEUE_URL;
  if (!configuredSubscription || !configuredAthlete || !queueUrl) {
    console.error("[webhook] not fully configured; event dropped");
    return acknowledge();
  }
  // These IDs are not secret; the worker confirms every event with Strava.
  if (
    String(event.subscription_id) !== configuredSubscription ||
    String(event.owner_id) !== configuredAthlete
  ) {
    console.warn("[webhook] event for another subscription or athlete dropped");
    return acknowledge();
  }

  try {
    await sqs.send(
      new SendMessageCommand({
        QueueUrl: queueUrl,
        MessageBody: JSON.stringify({
          object_type: event.object_type,
          aspect_type: event.aspect_type,
          object_id: event.object_id,
          owner_id: event.owner_id,
          subscription_id: event.subscription_id,
          event_time: event.event_time,
        }),
        MessageGroupId: EVENTS_MESSAGE_GROUP,
      }),
    );
  } catch (error) {
    console.error("[webhook] could not enqueue", error);
    return NextResponse.json({ error: "Could not queue event" }, { status: 503 });
  }

  return acknowledge();
}
