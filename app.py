from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for servers
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app) 

@app.route('/')
def home():
    return "Nuclear Reaction Simulation API is running!"

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        ma = data['ma'] * 931.5
        mX = data['mX'] * 931.5
        mb = data['mb'] * 931.5
        mY = data['mY'] * 931.5
        Q  = data['Q']
        Ta = data['Ta']

        a = ma / 931.5
        X = mX / 931.5
        b = mb / 931.5
        Y = mY / 931.5
        pa = np.sqrt(2 * ma * Ta)

        Tb_arr = []
        TY_arr = []
        theta_b_deg_arr = []
        theta_Y_deg_arr = []

        gamma = np.sqrt(((ma * mb) / (mX * mY)) * (Ta / (Ta + Q * (1 + ma / mX))))
        gamma_ = np.sqrt(((ma * mY) / (mX * mb)) * (Ta / (Ta + Q * (1 + ma / mX))))

        for _ in range(1000):
            theta_b_cm = np.random.uniform(0, np.pi)
            phi_b_cm = np.random.uniform(0, 2 * np.pi)

            theta_Y_cm = np.pi - theta_b_cm
            phi_Y_cm = (phi_b_cm + np.pi) % (2 * np.pi)

            theta_b = np.arctan2(np.sin(theta_b_cm), (np.cos(theta_b_cm) + gamma))
            theta_b = theta_b + np.pi if theta_b < 0 else theta_b
            theta_Y = np.arctan2(np.sin(theta_Y_cm), (np.cos(theta_Y_cm) + gamma_))
            theta_Y = theta_Y + np.pi if theta_Y < 0 else theta_Y

            Tb_1 = ((np.sqrt(ma * mb * Ta) * np.cos(theta_b) + 
                    np.sqrt(ma * mb * Ta * (np.cos(theta_b))**2 + 
                            (mY + mb) * (mY * Q + mY * Ta - ma * Ta))) / (mY + mb)) ** 2
            Tb_2 = ((np.sqrt(ma * mb * Ta) * np.cos(theta_b) - 
                    np.sqrt(ma * mb * Ta * (np.cos(theta_b))**2 + 
                            (mY + mb) * (mY * Q + mY * Ta - ma * Ta))) / (mY + mb)) ** 2

            pb_1 = np.sqrt(2 * mb * Tb_1)
            pb_2 = np.sqrt(2 * mb * Tb_2)

            phi_b = phi_b_cm

            pb_1_vec = np.array([
                pb_1 * np.sin(theta_b) * np.cos(phi_b),
                pb_1 * np.sin(theta_b) * np.sin(phi_b),
                pb_1 * np.cos(theta_b)
            ])
            pb_2_vec = np.array([
                pb_2 * np.sin(theta_b) * np.cos(phi_b),
                pb_2 * np.sin(theta_b) * np.sin(phi_b),
                pb_2 * np.cos(theta_b)
            ])

            pa_vec = np.array([0, 0, pa])
            pY_1_vec = pa_vec - pb_1_vec
            pY_2_vec = pa_vec - pb_2_vec

            theta_Y1 = np.arctan2(np.linalg.norm(pY_1_vec[:2]), pY_1_vec[2])
            theta_Y2 = np.arctan2(np.linalg.norm(pY_2_vec[:2]), pY_2_vec[2])

            pY_1_mag = np.linalg.norm(pY_1_vec)
            pY_2_mag = np.linalg.norm(pY_2_vec)

            TY_1 = (pY_1_mag ** 2) / (2 * mY)
            TY_2 = (pY_2_mag ** 2) / (2 * mY)

            Tb = Tb_1 if abs(theta_Y - theta_Y1) < abs(theta_Y - theta_Y2) else Tb_2
            TY = TY_1 if abs(theta_Y - theta_Y1) < abs(theta_Y - theta_Y2) else TY_2

            theta_b_deg_arr.append(np.degrees(theta_b))
            theta_Y_deg_arr.append(np.degrees(theta_Y))
            Tb_arr.append(Tb / mb)
            TY_arr.append(TY / mY)

        # Plotting
        plt.figure()
        plt.plot(theta_b_deg_arr, Tb_arr, 'o', markersize=3, label='b')
        plt.plot(theta_Y_deg_arr, TY_arr, 'o', markersize=3, label='Y')
        plt.ylabel("Energy [Lab-MeV/u]")
        plt.xlabel("Angle [Lab-deg]")
        plt.legend()
        plt.title("Kinematic Plot")
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return jsonify({"image": img_str})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
