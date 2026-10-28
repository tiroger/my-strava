import "server-only";

import {
  GetSecretValueCommand,
  SecretsManagerClient,
} from "@aws-sdk/client-secrets-manager";

// Strava OAuth credentials live in a separate secret that only the worker can read.
export type WebSecret = {
  webhook_verify_token: string;
  admin_passphrase: string;
  admin_session_secret: string;
};

const minimumLengths: Record<keyof WebSecret, number> = {
  webhook_verify_token: 16,
  admin_passphrase: 16,
  admin_session_secret: 32,
};

const client = new SecretsManagerClient({});
let cached: Promise<WebSecret> | undefined;

export function getWebSecret(): Promise<WebSecret> {
  if (!cached) {
    const secretArn = process.env.WEB_SECRET_ARN;
    if (!secretArn) throw new Error("WEB_SECRET_ARN is not configured");
    cached = client
      .send(new GetSecretValueCommand({ SecretId: secretArn }))
      .then((response) => {
        if (!response.SecretString) {
          throw new Error("Web secret has no SecretString");
        }
        const value = JSON.parse(response.SecretString) as unknown;
        if (typeof value !== "object" || value === null || Array.isArray(value)) {
          throw new Error("Web secret must contain a JSON object");
        }
        const secret = value as Record<string, unknown>;
        for (const [key, minimum] of Object.entries(minimumLengths)) {
          const field = secret[key];
          if (typeof field !== "string" || field.length < minimum) {
            throw new Error(
              `Web secret ${key} must be a string of at least ${minimum} characters`,
            );
          }
        }
        return secret as WebSecret;
      })
      .catch((error: unknown) => {
        cached = undefined;
        throw error;
      });
  }
  return cached;
}
