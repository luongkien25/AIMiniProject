/*
  Shopee issue-candidate collector.

  How to use:
  1. Open a Shopee product page and scroll to the review section.
  2. Prefer selecting 1-3 star filters when you need more issue reviews.
  3. Paste this whole file into DevTools Console.
  4. Run:
     collectShopeeIssueCandidatesByPage(1, 20)

  Output CSV columns:
  page, issue_candidate, score, matched_patterns, review_text
*/

const ISSUE_TARGETS = [
  "price_value",
  "seller_service",
  "shipping_delivery",
  "packaging",
  "wrong_missing_item",
  "spam_irrelevant",
];

const ISSUE_PATTERNS = {
  price_value: [
    "khong dang tien",
    "khong xung",
    "khong xung gia",
    "phi tien",
    "gia cao",
    "gia dat",
    "dat so voi",
    "chat luong khong xung",
    "tam gia nay khong on",
    "mua phi",
  ],
  seller_service: [
    "shop khong rep",
    "shop ko rep",
    "shop k rep",
    "shop im",
    "khong tra loi",
    "ko tra loi",
    "k tra loi",
    "khong phan hoi",
    "khong ho tro",
    "tu van sai",
    "shop tu van sai",
    "khong giai quyet",
    "thai do",
    "coc can",
    "noi chuyen te",
  ],
  shipping_delivery: [
    "giao cham",
    "giao hang cham",
    "giao lau",
    "giao qua lau",
    "ship cham",
    "ship lau",
    "van chuyen cham",
    "van chuyen lau",
    "doi hang lau",
    "ca tuan moi nhan",
    "that lac",
    "mat don",
    "shipper thai do",
  ],
  packaging: [
    "dong goi so sai",
    "dong goi khong can than",
    "dong goi khong chac",
    "dong goi te",
    "goi hang so sai",
    "goi so sai",
    "hop mop",
    "hop meo",
    "hop rach",
    "hop vo",
    "hop nat",
    "bao bi rach",
    "bao bi mop",
    "khong chong soc",
    "khong co chong soc",
  ],
  wrong_missing_item: [
    "giao sai",
    "giao nham",
    "giao thieu",
    "shop giao sai",
    "shop giao nham",
    "shop giao thieu",
    "thieu hang",
    "thieu mon",
    "thieu so luong",
    "thieu phu kien",
    "thieu day",
    "thieu sac",
    "sai mau",
    "sai size",
    "sai kich co",
    "khong du hang",
    "khong du so luong",
    "mua 2 giao 1",
    "mua hai giao mot",
    "dat mau",
    "dat size",
  ],
  spam_irrelevant: [
    "nhan xu",
    "lay xu",
    "kiem xu",
    "danh gia nhan xu",
    "cho du ky tu",
    "hinh anh khong lien quan",
    "video khong lien quan",
    "hinh anh chi mang tinh",
    "video chi mang tinh",
    "minh hoa",
  ],
};

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function normalizeVietnamese(text) {
  return String(text || "")
    .toLowerCase()
    .replaceAll("đ", "d")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\b(k|kh|ko|hong|hok|hông|hổng)\b/g, " khong ")
    .replace(/\b(dc|đc|dk|đk)\b/g, " duoc ")
    .replace(/\b(chx|cxua)\b/g, " chua ")
    .replace(/\b(sp)\b/g, " san pham ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function csvEscape(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function downloadCSV(filename, rows) {
  const header = [
    "page",
    "issue_candidate",
    "score",
    "matched_patterns",
    "review_text",
  ];

  const csv = [
    header.join(","),
    ...rows.map(row => [
      row.page,
      row.issue_candidate,
      row.score,
      csvEscape(row.matched_patterns.join("; ")),
      csvEscape(row.review_text),
    ].join(",")),
  ].join("\n");

  const blob = new Blob(["\uFEFF" + csv], {
    type: "text/csv;charset=utf-8;",
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function getReviewsFromVisibleText() {
  const lines = document.body.innerText
    .split("\n")
    .map(x => x.trim())
    .filter(Boolean);

  const reviews = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (/\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}/.test(line)) {
      const block = [];

      for (let j = i + 1; j < lines.length; j++) {
        const next = lines[j];

        if (/\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}/.test(next)) break;
        if (/Phản Hồi Của Người Bán/i.test(next)) break;
        if (/Hữu ích\??/i.test(next)) break;
        if (/^Phân loại hàng:/i.test(next)) continue;
        if (/^[★☆\s]+$/.test(next)) continue;
        if (/^\.\.\.$/.test(next)) continue;

        block.push(next);
      }

      const review = block.join(" ").replace(/\s+/g, " ").trim();
      if (review.length >= 3) reviews.push(review);
    }
  }

  return [...new Set(reviews)];
}

function classifyIssueCandidate(reviewText, targetIssues = ISSUE_TARGETS) {
  const normalized = normalizeVietnamese(reviewText);
  const candidates = [];

  for (const issue of targetIssues) {
    const patterns = ISSUE_PATTERNS[issue] || [];
    const matched = patterns.filter(pattern => normalized.includes(pattern));

    if (matched.length > 0) {
      candidates.push({
        issue_candidate: issue,
        score: matched.length,
        matched_patterns: matched,
      });
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  return candidates[0] || null;
}

function clickPageNumber(pageNumber) {
  const buttons = [...document.querySelectorAll("button, a")];
  const target = buttons.find(btn => {
    return (btn.innerText || "").trim() === String(pageNumber);
  });

  if (!target) return false;

  target.scrollIntoView({ block: "center" });
  target.click();
  return true;
}

async function collectShopeeIssueCandidatesByPage(
  startPage = 1,
  endPage = 20,
  options = {},
) {
  const targetIssues = options.targetIssues || ISSUE_TARGETS;
  const filename = options.filename || "shopee_issue_candidates.csv";
  const waitMs = options.waitMs || 2500;
  const minScore = options.minScore || 1;

  const all = [];
  const seen = new Set();
  const summary = Object.fromEntries(targetIssues.map(issue => [issue, 0]));

  for (let page = startPage; page <= endPage; page++) {
    console.log(`Dang chuyen toi trang ${page}...`);

    clickPageNumber(page);
    await sleep(waitMs);

    const reviews = getReviewsFromVisibleText();
    let keptOnPage = 0;

    for (const reviewText of reviews) {
      const normalized = normalizeVietnamese(reviewText);
      if (seen.has(normalized)) continue;

      const candidate = classifyIssueCandidate(reviewText, targetIssues);
      if (!candidate || candidate.score < minScore) continue;

      seen.add(normalized);
      keptOnPage += 1;
      summary[candidate.issue_candidate] += 1;

      all.push({
        page,
        review_text: reviewText,
        ...candidate,
      });
    }

    console.log(
      `Trang ${page}: thay ${reviews.length} review, giu ${keptOnPage} issue candidates`,
    );
  }

  console.table(summary);
  console.log(`Tong candidates: ${all.length}`);
  console.log(all);

  downloadCSV(filename, all);
}

// Example:
// collectShopeeIssueCandidatesByPage(1, 20)
// collectShopeeIssueCandidatesByPage(1, 20, {
//   targetIssues: ["price_value", "seller_service"],
//   filename: "shopee_price_seller_candidates.csv",
// })
