# 📄 مستندات API - دستیار هوشمند جلسات
### نسخه 1.0.0 | تاریخ: ۱ مرداد ۱۴۰۵

---

## 🔎 خلاصه سیستم

این سرویس فایل صوتی جلسات (WebM, MP4, WAV) را دریافت کرده و با استفاده از هوش مصنوعی محلی، خروجی JSON کاملی شامل موارد زیر تولید می‌کند:

| خروجی | توضیح |
|-------|-------|
| **رونوشت (transcript)** | متن کامل جلسه با تایم‌کد |
| **خلاصه (summary)** | خلاصه یک صفحه‌ای از محتوای جلسه |
| **نکات کلیدی (key_points)** | لیست نکات مهم بحث شده |
| **تصمیمات (decisions)** | تصمیمات اتخاذ شده با زمینه |
| **وظایف (action_items)** | تسک‌ها با مسئول، اولویت و مهلت |
| **بینش‌ها (insights)** | ریسک‌ها، فرصت‌ها و توصیه‌های مدیریتی |

---

## 🌐 آدرس پایه (Base URL)

```
http://localhost:8000
```

> مستندات تعاملی Swagger:  `http://localhost:8000/docs`

---

## 📡 Endpointها

---

### 1. `POST /api/v1/meetings/process` — پردازش فایل صوتی جلسه

**اصلی‌ترین endpoint.** فایل صوتی آپلود می‌شود و JSON کامل تحلیل جلسه برمی‌گردد.

**ورودی:**

| پارامتر | نوع | الزامی | توضیح |
|---------|------|--------|-------|
| `audio` | `File (multipart/form-data)` | ✅ بله | فایل صوتی جلسه (WebM, MP4, WAV, MP3, M4A, FLAC, OGG) |
| `meeting_id` | `string (query param)` | ❌ خیر | شناسه دلخواه برای جلسه. اگر ارسال نشود خودکار تولید می‌شود |

**نمونه درخواست با `curl`:**

```bash
curl -X POST "http://localhost:8000/api/v1/meetings/process" \
  -F "audio=@meeting_recording.webm"
```

**نمونه درخواست با `curl` + meeting_id سفارشی:**

```bash
curl -X POST "http://localhost:8000/api/v1/meetings/process?meeting_id=meet_2026_07_22" \
  -F "audio=@meeting_recording.webm"
```

**نمونه درخواست با JavaScript `fetch`:**

```javascript
const formData = new FormData();
formData.append('audio', audioFile); // audioFile = فایل WebM از WebRTC

const response = await fetch('http://localhost:8000/api/v1/meetings/process', {
    method: 'POST',
    body: formData,
});

const result = await response.json();
console.log(result);
```

**خروجی (Response) — `200 OK`:**

```json
{
  "meeting_id": "meeting_20260722_103000_a1b2c3",

  "transcript": [
    {
      "speaker": "Speaker",
      "start_time": "00:00.000",
      "end_time": "00:05.230",
      "text": "سلام و وقت بخیر، امروز می‌خواهیم درباره پروژه جدید صحبت کنیم."
    },
    {
      "speaker": "Speaker",
      "start_time": "00:05.230",
      "end_time": "00:12.500",
      "text": "بله، ما باید تا پایان ماه نسخه اول را آماده کنیم."
    }
  ],

  "summary": "در این جلسه درباره پروژه جدید بحث شد. تیم توسعه موظف به ارائه نسخه اول تا پایان ماه شد. همچنین ریسک‌های احتمالی کمبود نیرو بررسی و تصمیم به استخدام نیروی جدید گرفته شد.",

  "key_points": [
    "ارائه نسخه اول تا پایان ماه",
    "نیاز به استخدام نیروی جدید",
    "بررسی ریسک کمبود منابع"
  ],

  "decisions": [
    {
      "decision": "استخدام یک برنامه‌نویس بک‌اند جدید",
      "context": "به دلیل حجم کار زیاد و مهلت کوتاه پروژه"
    },
    {
      "decision": "استفاده از فریمورک FastAPI برای بک‌اند",
      "context": "به دلیل سرعت توسعه و مستندسازی خودکار"
    }
  ],

  "action_items": [
    {
      "task": "آماده‌سازی نسخه اول پروژه",
      "assignee": "تیم توسعه",
      "priority": "High",
      "deadline": "پایان مرداد ماه"
    },
    {
      "task": "انتشار آگهی استخدام",
      "assignee": "واحد منابع انسانی",
      "priority": "Normal",
      "deadline": null
    }
  ],

  "insights": {
    "risks": [
      "احتمال تأخیر در تحویل به دلیل کمبود نیروی انسانی",
      "وابستگی زیاد پروژه به یک نفر"
    ],
    "opportunities": [
      "استفاده از هوش مصنوعی برای افزایش بهره‌وری تیم",
      "امکان فروش محصول به سازمان‌های دیگر"
    ],
    "recommendations": [
      "تدوین برنامه زمان‌بندی دقیق برای هر فاز پروژه",
      "برگزاری جلسات هفتگی برای پیگیری پیشرفت"
    ]
  }
}
```

---

### 2. `POST /api/v1/meetings/chat` — چت هوشمند درباره جلسات

از این endpoint برای پرسیدن سوال درباره جلسات **قبلاً پردازش شده** استفاده کنید. سیستم RAG از متن جلسات قبلی جستجو کرده و پاسخ می‌دهد.

**ورودی:**

| پارامتر | نوع | الزامی | توضیح |
|---------|------|--------|-------|
| `question` | `string` | ✅ بله | سوال کاربر به زبان فارسی |

**Content-Type:** `application/json`

**نمونه درخواست با `curl`:**

```bash
curl -X POST "http://localhost:8000/api/v1/meetings/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "در جلسه قبلی چه تصمیماتی گرفته شد؟"}'
```

**نمونه درخواست با JavaScript `fetch`:**

```javascript
const response = await fetch('http://localhost:8000/api/v1/meetings/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: 'در جلسه قبلی چه تصمیماتی گرفته شد؟' }),
});

const result = await response.json();
console.log(result.answer);
```

**خروجی (Response) — `200 OK`:**

```json
{
  "answer": "بر اساس رونوشت جلسه [سند 1]، تصمیم گرفته شد که یک برنامه‌نویس بک‌اند جدید استخدام شود. همچنین [سند 2] نشان می‌دهد که فریمورک FastAPI برای توسعه بک‌اند انتخاب شده است.",
  "citations": [
    {
      "source_num": 1,
      "speaker": "Speaker",
      "start_time": "00:05.230",
      "end_time": "00:12.500",
      "text": "ما باید نیروی جدید استخدام کنیم...",
      "relevance_score": 0.892
    },
    {
      "source_num": 2,
      "speaker": "Speaker",
      "start_time": "00:30.100",
      "end_time": "00:38.200",
      "text": "برای بک‌اند از FastAPI استفاده می‌کنیم...",
      "relevance_score": 0.847
    }
  ]
}
```

---

### 3. `GET /api/v1/health` — بررسی سلامت سرویس

از این endpoint برای بررسی اینکه سرویس AI بالاست و مدل‌ها لود شده‌اند استفاده کنید.

**ورودی:** ندارد

**نمونه درخواست:**

```bash
curl http://localhost:8000/api/v1/health
```

**خروجی (Response) — `200 OK`:**

```json
{
  "status": "ok",
  "whisper_model": "loaded",
  "ollama_status": "connected"
}
```

| فیلد | مقادیر ممکن | توضیح |
|------|-------------|-------|
| `status` | `"ok"` | وضعیت کلی سرویس |
| `whisper_model` | `"loaded"` / `"not_loaded"` | آیا مدل Whisper لود شده |
| `ollama_status` | `"connected"` / `"disconnected"` | آیا اتصال به Ollama برقرار است |

---

## ⚠️ کدهای خطا

| کد HTTP | معنی | مثال |
|---------|------|------|
| `400` | ورودی نامعتبر | فرمت فایل پشتیبانی نمی‌شود / فایل خالی |
| `404` | یافت نشد | فایل صوتی پیدا نشد |
| `422` | خطای پردازش | متنی از فایل صوتی استخراج نشد |
| `500` | خطای سرور | مشکل در اتصال به Ollama یا خطای داخلی |
| `503` | سرویس آماده نیست | مدل‌ها هنوز لود نشده‌اند |

**فرمت خطا:**

```json
{
  "detail": "Unsupported file format: .txt. Allowed: .webm, .mp4, .wav, .mp3, .m4a, .flac, .ogg"
}
```

---

## 📋 فرمت‌های صوتی پشتیبانی شده

| فرمت | پسوند فایل |
|------|-----------|
| WebM | `.webm` |
| MP4 | `.mp4` |
| WAV | `.wav` |
| MP3 | `.mp3` |
| M4A | `.m4a` |
| FLAC | `.flac` |
| OGG | `.ogg` |

---

## 🔧 نکات فنی برای فرانت‌اند

1. **آپلود فایل:** حتماً از `multipart/form-data` استفاده کنید (نه `application/json`)
2. **نام فیلد:** نام فیلد فایل باید **`audio`** باشد
3. **سایز فایل:** محدودیت خاصی نیست ولی فایل‌های بزرگ‌تر زمان پردازش بیشتری می‌برند
4. **زمان پردازش:** بسته به طول فایل صوتی، پردازش ممکن است **۱ تا ۱۰ دقیقه** طول بکشد (شامل ASR + تحلیل)
5. **CORS:** تمام originها مجاز هستند (در حال حاضر `*`)
6. **Encoding:** تمام پاسخ‌ها UTF-8 هستند

---

## 🏗️ معماری سیستم

```
فرانت‌اند (رضا)  ──►  FastAPI (محمدرضا)  ──►  مدل‌های AI محلی
     │                      │                       │
     │   WebM آپلود         │   Whisper large-v3    │  رونویسی صدا
     │                      │   DeepSeek-R1:8b      │  تصحیح متن
     │                      │   Qwen2.5:7b          │  خلاصه + تحلیل
     │                      │                       │
     │◄── JSON پاسخ ────────│                       │
```

---

## 📌 نمونه کامل ادغام با React/Next.js

```javascript
// components/MeetingUpload.jsx

async function processMeeting(audioBlob) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'meeting.webm');

  try {
    const response = await fetch('http://localhost:8000/api/v1/meetings/process', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail);
    }

    const result = await response.json();
    
    // result.meeting_id     → شناسه جلسه
    // result.transcript     → رونوشت کامل
    // result.summary        → خلاصه جلسه
    // result.key_points     → نکات کلیدی
    // result.decisions      → تصمیمات
    // result.action_items   → وظایف
    // result.insights       → بینش‌های مدیریتی
    
    return result;
  } catch (error) {
    console.error('Error processing meeting:', error);
    throw error;
  }
}

// استفاده از چت RAG
async function askQuestion(question) {
  const response = await fetch('http://localhost:8000/api/v1/meetings/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });

  const result = await response.json();
  return result; // { answer: "...", citations: [...] }
}
```
