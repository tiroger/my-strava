import { Dashboard } from "@/components/dashboard";
import { isAdminSessionValid } from "@/lib/admin-session";
import { loadDashboardData } from "@/lib/activities";
import { localIsoDate } from "@/lib/activity-types";

export const dynamic = "force-dynamic";

async function adminSession(): Promise<boolean> {
  try {
    return await isAdminSessionValid();
  } catch (error) {
    console.error("Could not verify admin session", error);
    return false;
  }
}

export default async function Home() {
  const [data, isAdmin] = await Promise.all([loadDashboardData(), adminSession()]);
  return (
    <Dashboard
      dataVersion={data.version}
      initialActivities={data.activities}
      initialAdmin={isAdmin}
      initialGear={data.gear}
      initialMaintenance={data.maintenance}
      initialSettings={data.settings}
      serverToday={localIsoDate()}
    />
  );
}
