class ClinicalLogic:
    @staticmethod
    def get_treatment_plan(predicted_class):
        """
        Returns a clinical recommendation dictionary based on the predicted class.
        Classes: ['glioma', 'meningioma', 'notumor', 'pituitary'] (based on dataset folder names usually)
        """
        # Normalize class name
        cls = predicted_class.lower().replace("_", "").replace(" ", "")
        
        if "notumor" in cls or "no" in cls:
            return {
                "Risk Level": "Low",
                "Diagnosis": "No Tumor Detected",
                "Immediate Actions": "None required.",
                "Follow-up": "Routine check-up in 12 months."
            }
        elif "glioma" in cls:
            return {
                "Risk Level": "High",
                "Diagnosis": "Glioma Detected",
                "Immediate Actions": "Urgent Oncology Referral. Schedule MRI with contrast.",
                "Follow-up": "Biopsy required for grading."
            }
        elif "meningioma" in cls:
            return {
                "Risk Level": "Medium",
                "Diagnosis": "Meningioma Detected",
                "Immediate Actions": "Neurosurgery Consultation.",
                "Follow-up": "Monitor growth. Surgical resection may be needed."
            }
        elif "pituitary" in cls:
            return {
                "Risk Level": "Medium",
                "Diagnosis": "Pituitary Tumor Detected",
                "Immediate Actions": "Endocrinology Referral. Check hormone levels.",
                "Follow-up": "MRI Sella Turcica protocol."
            }
        else:
            return {
                "Risk Level": "Unknown",
                "Diagnosis": f"Unknown Class: {predicted_class}",
                "Immediate Actions": "Manual Review Required.",
                "Follow-up": "N/A"
            }
