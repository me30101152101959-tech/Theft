"""
AI Copilot Router — /api/copilot
Provides XAI explanations in Arabic and English.
Uses rule-based reasoning from real prediction data.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from models.schemas import CopilotRequest
from services import dataset_service, model_service

router = APIRouter(prefix="/api/copilot", tags=["copilot"])

# ─────────────────────────────────────────────
# XAI explanation engine
# ─────────────────────────────────────────────
def explain_prediction(c: dict, lang: str = "en") -> str:
    prob = c["probability"]
    risk = c["risk_score"]
    conf = c["confidence"]
    status = c["status"]
    readings = c.get("readings", [])

    # Compute signal features for explanation
    import numpy as np
    r = np.array(readings) if readings else np.array([0] * 26)
    diff = np.diff(r)
    volatility = float(np.std(diff))
    mean_r = float(np.mean(r))
    max_r = float(np.max(r))
    min_r = float(np.min(r))
    zero_count = int(np.sum(r == 0))
    negative_count = int(np.sum(r < 0))
    
    h1_mean = float(np.mean(r[:13]))
    h2_mean = float(np.mean(r[13:]))
    trend_ratio = h2_mean / (h1_mean + 1e-8)

    if lang == "ar":
        if status == "Theft":
            explanation = (
                f"🚨 **تشخيص: سرقة كهرباء**\n\n"
                f"**احتمالية السرقة:** {prob*100:.1f}%\n"
                f"**درجة المخاطرة:** {risk:.1f}/100\n"
                f"**مستوى الثقة:** {conf*100:.1f}%\n\n"
                f"**الأسباب التي أدت إلى هذا التشخيص:**\n"
            )
            if volatility > mean_r * 0.5:
                explanation += "• تذبذب عالٍ في القراءات مما يشير إلى نمط غير طبيعي.\n"
            if zero_count > 3:
                explanation += f"• عدد القراءات الصفرية: {zero_count} — وهو مؤشر على التلاعب.\n"
            if negative_count > 0:
                explanation += f"• وجود {negative_count} قراءة سالبة — لا يحدث في الاستهلاك الطبيعي.\n"
            if trend_ratio < 0.6:
                explanation += "• انخفاض ملحوظ في الاستهلاك في النصف الثاني — نمط مشبوه.\n"
            if trend_ratio > 2.0:
                explanation += "• ارتفاع مفاجئ غير مبرر في الاستهلاك.\n"
            explanation += (
                f"\n**توصية:** يُنصح بإرسال فريق تفتيش فوري للعميل.\n"
                f"النموذج المستخدم: {model_service.get_model_info().get('model_name', 'CNN-LSTM')}"
            )
        else:
            explanation = (
                f"✅ **تشخيص: عميل طبيعي**\n\n"
                f"**احتمالية السرقة:** {prob*100:.1f}%\n"
                f"**درجة المخاطرة:** {risk:.1f}/100\n"
                f"**مستوى الثقة:** {conf*100:.1f}%\n\n"
                f"**ملاحظات:**\n"
                f"• نمط الاستهلاك منتظم وضمن الحدود المعتادة.\n"
                f"• متوسط الاستهلاك: {mean_r:.1f} وحدة.\n"
                f"• لا توجد مؤشرات على التلاعب في البيانات.\n\n"
                f"النموذج المستخدم: {model_service.get_model_info().get('model_name', 'CNN-LSTM')}"
            )
    else:
        if status == "Theft":
            explanation = (
                f"🚨 **Diagnosis: Electricity Theft Detected**\n\n"
                f"**Theft Probability:** {prob*100:.1f}%\n"
                f"**Risk Score:** {risk:.1f}/100\n"
                f"**Confidence:** {conf*100:.1f}%\n\n"
                f"**Key Factors Contributing to This Decision:**\n"
            )
            if volatility > mean_r * 0.5:
                explanation += f"• High reading volatility (σ={volatility:.1f}) indicates an abnormal consumption pattern.\n"
            if zero_count > 3:
                explanation += f"• {zero_count} zero readings detected — possible meter tampering.\n"
            if negative_count > 0:
                explanation += f"• {negative_count} negative reading(s) found — physically impossible in honest consumption.\n"
            if trend_ratio < 0.6:
                explanation += "• Significant consumption drop in the second half — suspicious pattern.\n"
            if trend_ratio > 2.0:
                explanation += "• Unexplained consumption surge in the second half.\n"
            explanation += (
                f"\n**Recommendation:** Dispatch an inspection team immediately.\n"
                f"Model: {model_service.get_model_info().get('model_name', 'CNN-LSTM')}"
            )
        else:
            explanation = (
                f"✅ **Diagnosis: Normal Customer**\n\n"
                f"**Theft Probability:** {prob*100:.1f}%\n"
                f"**Risk Score:** {risk:.1f}/100\n"
                f"**Confidence:** {conf*100:.1f}%\n\n"
                f"**Analysis:**\n"
                f"• Consumption pattern is regular and within expected bounds.\n"
                f"• Average consumption: {mean_r:.1f} units.\n"
                f"• No manipulation indicators detected.\n\n"
                f"Model: {model_service.get_model_info().get('model_name', 'CNN-LSTM')}"
            )

    return explanation


def answer_general_question(question: str, lang: str) -> str:
    q = question.lower()
    stats = dataset_service.get_dashboard_stats()

    # FAQ matching
    if any(w in q for w in ["accuracy", "دقة"]):
        acc = stats.get("accuracy", None)
        if acc:
            return f"Model Accuracy: {acc*100:.2f}%" if lang == "en" else f"دقة النموذج: {acc*100:.2f}%"
        return "Accuracy requires a labelled dataset (FLAG column)." if lang == "en" else "تحتاج إلى عمود FLAG لحساب الدقة."

    if any(w in q for w in ["theft", "سرقة", "how many"]):
        t = stats.get("predicted_theft", 0)
        total = stats.get("total_customers", 0)
        return (
            f"Detected {t:,} theft cases out of {total:,} customers ({t/total*100:.1f}%)."
            if lang == "en" else
            f"تم اكتشاف {t:,} حالة سرقة من إجمالي {total:,} عميل ({t/total*100:.1f}%)."
        )

    if any(w in q for w in ["model", "نموذج", "architecture", "معمارية"]):
        info = model_service.get_model_info()
        return (
            f"Loaded model: {info.get('model_name')} | "
            f"Architecture: CNN-LSTM | "
            f"Parameters: {info.get('total_params_fmt')} | "
            f"Input: {info.get('input_shape')}"
            if lang == "en" else
            f"النموذج المحمّل: {info.get('model_name')} | "
            f"المعمارية: CNN-LSTM | "
            f"المعاملات: {info.get('total_params_fmt')}"
        )

    if any(w in q for w in ["risk", "مخاطر", "high risk"]):
        customers = dataset_service.get_all_customers_for_export()
        if customers:
            top = sorted(customers, key=lambda x: x["risk_score"], reverse=True)[:5]
            names = ", ".join(c["id"][:8] + "…" for c in top)
            return (
                f"Top 5 highest-risk customers: {names}"
                if lang == "en" else
                f"أعلى 5 عملاء مخاطرةً: {names}"
            )

    if any(w in q for w in ["confidence", "ثقة"]):
        avg = stats.get("avg_confidence", 0)
        return (
            f"Average model confidence: {avg*100:.2f}%"
            if lang == "en" else
            f"متوسط ثقة النموذج: {avg*100:.2f}%"
        )

    # Default
    return (
        "I can explain predictions, model metrics, risk scores, and detection results. "
        "Ask me about a specific customer or the overall statistics."
        if lang == "en" else
        "يمكنني شرح التنبؤات ومقاييس النموذج ودرجات المخاطر ونتائج الكشف. "
        "اسألني عن عميل محدد أو الإحصائيات العامة."
    )


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@router.post("/ask")
async def ask_copilot(req: CopilotRequest):
    lang = req.language if req.language in ("en", "ar") else "en"

    if req.customer_id:
        customer = dataset_service.get_customer_by_id(req.customer_id)
        if not customer:
            raise HTTPException(404, f"Customer {req.customer_id} not found.")
        answer = explain_prediction(customer, lang)
    else:
        answer = answer_general_question(req.question, lang)

    return JSONResponse({"success": True, "answer": answer, "language": lang})


@router.get("/explain/{customer_id}")
async def explain_customer(customer_id: str, lang: str = "en"):
    customer = dataset_service.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(404, f"Customer {customer_id} not found.")
    explanation = explain_prediction(customer, lang)
    return JSONResponse({"explanation": explanation, "customer": customer})
