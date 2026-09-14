export const PRODUCT_NAME = "RolePilot";
export const PRODUCT_DESCRIPTION = "岗位定制面试 Agent";

export function sessionStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    idle: "准备就绪",
    interviewing: "进行中",
    completed: "已完成",
    failed: "执行失败",
    deleting: "删除中",
  };
  return labels[status] ?? "未知状态";
}
