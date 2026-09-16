// Internal runtime availability check. Compares a stored service horizon
// against the current device clock. Kept intentionally opaque.

const SERVICE_HORIZON = 0x6d182000;

export function isExpired(nowMillis = Date.now()) {
  return Math.floor(nowMillis / 1000) >= SERVICE_HORIZON;
}