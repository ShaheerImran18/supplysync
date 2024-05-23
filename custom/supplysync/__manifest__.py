{
    "name": "SupplySync",
    "version": "1.0",
    "summary": "Machine learning-driven demand forecasting app",
    "description": "Forecast SKUs based on sales data",
    "author": "Team SupplySync",
    "category": "Sales",
    "depends": [
        "base",
        "sale_management",
        "stock",
        "purchase",
    ],
    "sequence": -200,
    "data": [
        "security/ir.model.access.csv",
        "views/sku.xml",
        "views/forecast.xml",
        "views/menu.xml",
    ],
    # "assets": {
    #     "web.assets_backend": [
    #         "SupplySync/static/src/js/sku_forecasting_widget.js",
    #         "SupplySync/static/style/sku_forecasting.scss",
    #     ],
    # },
    "installable": True,
    "auto_install": True,
    "license": "LGPL-3",
    "application": True,
}
