import "server-only";

import { createHmac } from "node:crypto";
import { cookies } from "next/headers";
import { getWebSecret } from "@/lib/secrets";
import { safeEqual } from "@/lib/security";

const COOKIE_NAME = "my-strava-admin";
const SESSION_TTL_SECONDS = 8 * 60 * 60;

function signature(expiresAt: string, secret: string): string {
  return createHmac("sha256", secret).update(expiresAt).digest("base64url");
}

export async function verifyAdminPassphrase(passphrase: string): Promise<boolean> {
  const secret = await getWebSecret();
  return safeEqual(passphrase, secret.admin_passphrase);
}

export async function createAdminCookie(): Promise<{
  value: string;
  maxAge: number;
}> {
  const secret = await getWebSecret();
  const maxAge = SESSION_TTL_SECONDS;
  const expiresAt = String(Math.floor(Date.now() / 1000) + maxAge);
  return {
    value: `${expiresAt}.${signature(expiresAt, secret.admin_session_secret)}`,
    maxAge,
  };
}

export async function isAdminSessionValid(): Promise<boolean> {
  const jar = await cookies();
  const value = jar.get(COOKIE_NAME)?.value;
  if (!value) return false;
  const [expiresAt, suppliedSignature, ...extra] = value.split(".");
  if (
    !expiresAt ||
    !suppliedSignature ||
    extra.length > 0 ||
    !/^\d+$/.test(expiresAt) ||
    Number(expiresAt) <= Math.floor(Date.now() / 1000)
  ) {
    return false;
  }
  const secret = await getWebSecret();
  return safeEqual(
    suppliedSignature,
    signature(expiresAt, secret.admin_session_secret),
  );
}

export const adminCookieName = COOKIE_NAME;
export const adminSessionTtlSeconds = SESSION_TTL_SECONDS;
