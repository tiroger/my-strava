import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const send = vi.hoisted(() => vi.fn());

vi.mock("@aws-sdk/client-sqs", () => ({
  SQSClient: class {
    send = send;
  },
  SendMessageCommand: class {
    constructor(readonly input: unknown) {}
  },
}));

const { POST } = await import("./route");

const event = {
  object_type: "activity",
  aspect_type: "create",
  object_id: 123,
  owner_id: 42,
  subscription_id: 7,
  event_time: 1_700_000_000,
};

const post = (body: unknown) =>
  POST(
    new Request("https://dashboard.example/api/strava/webhook", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  );

describe("Strava webhook", () => {
  beforeEach(() => {
    vi.stubEnv("STRAVA_SUBSCRIPTION_ID", "7");
    vi.stubEnv("STRAVA_ATHLETE_ID", "42");
    vi.stubEnv("STRAVA_EVENTS_QUEUE_URL", "https://sqs.example/queue.fifo");
    send.mockReset();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("queues recognized events in the shared FIFO message group", async () => {
    const response = await post({ ...event, updates: { title: "ignored" } });
    expect(response.status).toBe(200);
    expect(send).toHaveBeenCalledOnce();
    const input = send.mock.calls[0]?.[0].input;
    expect(input.MessageGroupId).toBe("strava-events");
    expect(JSON.parse(input.MessageBody)).toEqual(event);
  });

  it("acknowledges and drops events for another athlete or subscription", async () => {
    expect((await post({ ...event, owner_id: 1 })).status).toBe(200);
    expect((await post({ ...event, subscription_id: 1 })).status).toBe(200);
    expect(send).not.toHaveBeenCalled();
  });

  it("acknowledges athlete events without queueing them", async () => {
    expect((await post({ ...event, object_type: "athlete" })).status).toBe(200);
    expect(send).not.toHaveBeenCalled();
  });

  it("rejects malformed activity events", async () => {
    expect((await post({ ...event, object_id: "123" })).status).toBe(400);
  });

  it("acknowledges and drops events until the webhook is configured", async () => {
    vi.stubEnv("STRAVA_SUBSCRIPTION_ID", "");
    expect((await post(event)).status).toBe(200);
    expect(send).not.toHaveBeenCalled();
  });

  it("asks Strava to retry when the queue is unavailable", async () => {
    send.mockRejectedValueOnce(new Error("throttled"));
    expect((await post(event)).status).toBe(503);
  });
});
