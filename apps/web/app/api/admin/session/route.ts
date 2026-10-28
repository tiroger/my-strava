import {
  adminCookieName,
  createAdminCookie,
  isAdminSessionValid,
  verifyAdminPassphrase,
} from "@/lib/admin-session";
import { sameOrigin } from "@/lib/security";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ authenticated: await isAdminSessionValid() });
}

export async function POST(request: Request) {
  if (!sameOrigin(request)) {
    return NextResponse.json({ error: "Same-origin request required" }, { status: 403 });
  }

  let passphrase: unknown;
  try {
    ({ passphrase } = await request.json());
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }
  if (typeof passphrase !== "string" || passphrase.length < 1 || passphrase.length > 256) {
    return NextResponse.json({ error: "Invalid passphrase" }, { status: 400 });
  }
  if (!(await verifyAdminPassphrase(passphrase))) {
    return NextResponse.json({ error: "Incorrect passphrase" }, { status: 401 });
  }

  const session = await createAdminCookie();
  const response = NextResponse.json({ authenticated: true });
  response.cookies.set(adminCookieName, session.value, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    path: "/",
    maxAge: session.maxAge,
  });
  return response;
}

export async function DELETE(request: Request) {
  if (!sameOrigin(request)) {
    return NextResponse.json({ error: "Same-origin request required" }, { status: 403 });
  }
  const response = NextResponse.json({ authenticated: false });
  response.cookies.set(adminCookieName, "", {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    path: "/",
    maxAge: 0,
  });
  return response;
}
