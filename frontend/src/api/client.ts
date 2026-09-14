import type {
  AnswerRequest,
  AnswerResult,
  SessionPayload,
  SessionSummary,
  UploadResult,
} from "../types";

const CREDENTIAL_KEY = "interview_anonymous_bearer_v1";
let bootstrap: Promise<string> | null = null;
const ERROR_MESSAGES: Record<string, string> = {
  session_budget_exhausted: "当前会话无法继续生成，请结束并查看已有结果。",
  stale_question_version: "题目已更新，旧题回答未被评分；请刷新题目后重新作答。",
  stale_question: "题目已更新，旧题回答未被评分；请刷新题目后重新作答。",
  question_version_conflict: "题目版本不一致，请刷新后作答。",
  session_busy: "会话正在处理其他请求；原回答已保留，请稍后恢复原请求。",
  idempotency_conflict: "同一请求 ID 的内容不一致；已保留原回答，不能改内容后重发。",
  recovery_required: "回答处理需要服务端恢复；原请求已保留。可恢复原回答，或保留故障会话后开始新面试。",
  session_not_answerable: "当前面试已结束或不可继续，请查看最新会话状态。",
  session_completed: "面试已结束，请查看最新报告。",
  session_not_found: "当前凭证下找不到该会话；请确认没有清除或更换匿名凭证。",
  answer_request_not_found: "服务端尚未记录该回答请求，可恢复原回答重试。",
  operation_failed: "服务暂时无法完成操作，请稍后重试。",
  invalid_request: "请求内容不符合要求，请检查输入。",
  upload_too_large: "上传文件过大，单个文件最多 8 MiB。",
  unsupported_upload_format: "格式不支持：简历使用 PDF、DOCX、TXT 或图片，岗位图片使用常见图片格式。",
  resume_parse_failed: "简历识别失败，请检查文件或尝试 TXT 格式。",
  jd_image_parse_failed: "岗位图片识别失败，请换清晰图片或直接粘贴文字。",
  no_readable_text: "没有识别到可读文字，请换清晰图片或直接粘贴文字。",
  ocr_language_missing: "OCR 缺少所需中文或英文语言包，请管理员安装 chi_sim、eng 后重试；不会自动退回英文识别。",
  ocr_tesseract_missing: "未找到 Tesseract，请管理员安装 OCR 引擎或检查 TESSERACT_CMD。",
  ocr_tessdata_missing: "OCR 语言目录不存在，请管理员检查 OCR_TESSDATA_DIR。",
  ocr_invalid_languages: "OCR 语言配置无效，请管理员检查 OCR_LANGUAGES。",
  ocr_language_check_failed: "OCR 语言依赖自检失败，请管理员检查 Tesseract 安装。",
  ocr_pdf_renderer_missing: "扫描 PDF 渲染依赖不可用，请管理员检查 PDFium 或 Poppler 配置。",
  ocr_timeout: "OCR 识别超时，请减少页面或换更清晰的图片后重试。",
  ocr_failed: "OCR 引擎识别失败，请换清晰图片或使用文字版文件。",
  pdf_render_failed: "扫描 PDF 渲染失败，请检查文件或上传清晰图片。",
  ocr_image_too_large: "OCR 渲染图片过大，请降低页面尺寸或分开上传。",
};

function errorCode(detail: unknown): string {
  if (typeof detail === "string") return detail;
  const value = detail as { code?: string; status?: string } | null;
  return value?.code || value?.status || "";
}

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    const value = detail as { message?: string; code?: string; status?: string } | null;
    super(ERROR_MESSAGES[errorCode(detail)] || (status === 401
      ? "匿名凭证缺失或无效；为避免丢失会话，系统不会自动更换凭证。"
      : typeof detail === "string" ? detail : value?.message || `请求失败（${status}）`));
  }
  get code(): string {
    return errorCode(this.detail);
  }
}

async function createCredential(): Promise<string> {
  const existing = localStorage.getItem(CREDENTIAL_KEY);
  if (existing) return existing;
  const response = await fetch("/api/auth/anonymous", { method: "POST", cache: "no-store" });
  if (!response.ok) throw new ApiError(response.status, "匿名凭证创建失败，请重试");
  const credential = await response.json() as { token: string; token_type: string };
  if (!/^[A-Za-z0-9_-]{43}$/.test(credential.token) || credential.token_type !== "Bearer") {
    throw new Error("匿名凭证响应无效");
  }
  localStorage.setItem(CREDENTIAL_KEY, credential.token);
  return credential.token;
}

export async function anonymousCredential(): Promise<string> {
  const existing = localStorage.getItem(CREDENTIAL_KEY);
  if (existing) return existing;
  if (!bootstrap) {
    // Coordinate first visits across same-origin tabs where Web Locks is supported.
    bootstrap = (async () => {
      if (navigator.locks) {
        return await navigator.locks.request("interview-anonymous-bootstrap", createCredential);
      }
      return await createCredential();
    })().finally(() => { bootstrap = null; });
  }
  return bootstrap;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function once<T>(path: string, init: RequestInit): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${await anonymousCredential()}`);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...init, headers, cache: "no-store" });
  if (!response.ok) {
    let detail: unknown = `请求失败（${response.status}）`;
    try {
      const payload = await response.json();
      if (payload?.detail) detail = payload.detail;
      else if (payload?.status === "recovery_required") detail = payload;
    } catch {
      /* keep the default message */
    }
    // Never silently replace an invalid credential with a different owner.
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

/**
 * 读请求在网络抖动时自动重试一次；写请求（提交回答/开始面试）绝不自动重试，
 * 避免重复生成题目或重复计分。
 */
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = (init.method || "GET").toUpperCase();
  const attempts = method === "GET" ? 2 : 1;
  let lastError: unknown = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await once<T>(path, init);
    } catch (error) {
      lastError = error;
      const isNetworkError = error instanceof TypeError;
      if (isNetworkError && attempt + 1 < attempts) {
        await sleep(500);
        continue;
      }
      if (isNetworkError) {
        throw new Error("网络异常，请检查网络后重试；当前会话与输入已保留。");
      }
      throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("请求失败");
}

export const api = {
  uploadResume(file: File): Promise<UploadResult> {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResult>("/api/uploads/resume", { method: "POST", body: form });
  },

  uploadJdImage(file: File): Promise<UploadResult> {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResult>("/api/uploads/jd-image", { method: "POST", body: form });
  },

  startSession(body: {
    jd_text: string;
    resume_text: string;
    session_name: string;
  }): Promise<SessionPayload> {
    return request<SessionPayload>("/api/sessions", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  answer(sessionId: string, body: AnswerRequest): Promise<AnswerResult> {
    return request<AnswerResult>(`/api/sessions/${sessionId}/answer`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  answerStatus(sessionId: string, requestId: string): Promise<AnswerResult> {
    return request<AnswerResult>(`/api/sessions/${sessionId}/answer-requests/${requestId}`);
  },

  stop(sessionId: string): Promise<SessionPayload> {
    return request<SessionPayload>(`/api/sessions/${sessionId}/stop`, {
      method: "POST",
    });
  },

  /** P1/A1：修正系统识别到的岗位名；未答题时可顺带重新出题。 */
  updateJobTitle(
    sessionId: string,
    jobTitle: string,
    regenerate = true,
  ): Promise<SessionPayload> {
    return request<SessionPayload>(`/api/sessions/${sessionId}/job-title`, {
      method: "PATCH",
      body: JSON.stringify({ job_title: jobTitle, regenerate }),
    });
  },

  listSessions(): Promise<SessionSummary[]> {
    return request<SessionSummary[]>("/api/sessions");
  },

  getSession(sessionId: string): Promise<SessionPayload> {
    return request<SessionPayload>(`/api/sessions/${sessionId}`);
  },

  deleteSession(sessionId: string): Promise<{ deleted: boolean }> {
    return request<{ deleted: boolean }>(`/api/sessions/${sessionId}`, {
      method: "DELETE",
    });
  },

  renameSession(sessionId: string, name: string): Promise<SessionSummary> {
    return request<SessionSummary>(`/api/sessions/${sessionId}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    });
  },
};
