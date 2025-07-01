import forecast_model
import app
import sensebox

if __name__ == "__main__":
    app.app.run(debug=True, host='0.0.0.0', port=8050)
    # sensebox.run_loop()