"""Model card content, filled in from the artifact metadata."""


def build_model_card(meta):
    m = meta["metrics"]
    return {
        "modelDetails": {
            "name": meta["model_name"],
            "version": meta["model_version"],
            "trainedAt": meta["training_timestamp"],
            "hyperparameters": meta["hyperparameters"],
            "threshold": meta["threshold"],
        },
        "intendedUse": "Educational churn prediction and retention prioritisation. Helps retention "
                       "teams decide which customers to contact first.",
        "outOfScope": [
            "Autonomous customer cancellation, penalties or pricing decisions",
            "Treating a score as a guarantee that a customer will churn",
            "Use on customer populations outside the training data",
        ],
        "data": f"IBM Telco Customer Churn ({meta['dataset_version']}): 7,043 customers, "
                "19 features, about 26.5% churn.",
        "splitAndValidation": "Stratified 80/20 split. The test set was used once, after model, "
                              "hyperparameters and threshold were fixed. Random Search ran inside "
                              f"nested CV ({meta['cv_config']['outer']} outer, {meta['cv_config']['inner']} inner).",
        "evaluation": {
            "baselineCvF1": m["baseline_cv_f1"],
            "nestedCvF1": m["nested_cv_f1_mean"],
            "nestedCvF1Std": m["nested_cv_f1_std"],
            "testF1": m["test"]["f1"],
            "testBaselineF1": m["test_baseline"]["f1"],
            "testPrecision": m["test"]["precision"],
            "testRecall": m["test"]["recall"],
        },
        "limitations": [
            "The dataset may not represent today's or another company's customers.",
            "Historical churn labels can contain business-process artifacts.",
            "Distribution shift over time can reduce performance.",
            "Changing the threshold changes precision, recall and outreach workload.",
            "Feature importance shows association, not cause.",
        ],
        "ethicalConsiderations": "Scores are estimates. Use them with human review, never to deny "
                                 "service or treat customers unfairly. Monitor performance across "
                                 "customer groups.",
        "monitoring": "Track input drift, missing values, prediction distribution, calibration and "
                      "F1/precision/recall once real outcomes arrive.",
    }
