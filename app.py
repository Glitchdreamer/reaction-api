# === app.py ===
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from pynucastro import Nucleus
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, csv

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Three-body simulation API is running!"

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json

        a = Nucleus(data['a'])
        X = Nucleus(data['X'])
        b = Nucleus(data['b'])
        Y = Nucleus(data['Y'])
        
        ma = a.mass   
        mX = X.mass 
        mb = b.mass 
        mY = Y.mass 

        
        Ex_b  = data['Ex_b']
        Ex_Y  = data['Ex_Y']
        Ta = data['Ta']
        breakup = data.get('breakup', False)
        if breakup:
            c=Nucleus(data['c'])
            d=Nucleus(data['d'])
            mc = c.mass   # Convert to MeV
            md = d.mass 
        else:
            mc = 0
            md = 0
        Q_break = Ex_b + (mb - mc - md)
        Q = ((ma + mX) - (mb + mY)) - Ex_b - Ex_Y
        Ecm = mX / (ma + mX) * Ta
        vcb = (1.44 * a.Z * X.Z) / (1.2 * ((a.A)**(1/3) + (X.A)**(1/3)))
        n = 1000
        pa = np.sqrt(2 * ma * Ta)
        gamma = np.sqrt(((ma * mb) / (mX * mY)) * (Ta / (Ta + Q * (1 + ma / mX))))
        gamma_ = np.sqrt(((ma * mY) / (mX * mb)) * (Ta / (Ta + Q * (1 + ma / mX))))

        rows = [['theta_b_deg', 'Tb', 'theta_Y_deg', 'TY', 'theta_c', 'Tc', 'theta_d', 'Td', 'E_rel']]
        Tb_arr, TY_arr, theta_b_deg_arr, theta_Y_deg_arr = [], [], [], []

        for _ in range(n):
            theta_b_cm=np.random.uniform(0,np.pi)
            phi_b_cm=np.random.uniform(0,2*np.pi)

            theta_Y_cm=np.pi-theta_b_cm
            phi_Y_cm=phi_b_cm+np.pi
            if phi_Y_cm > 2*np.pi:
                phi_Y_cm -= 2*np.pi

            theta_b=np.arctan2(np.sin(theta_b_cm),(np.cos(theta_b_cm)+gamma))
            if theta_b < 0:
                theta_b += np.pi
            theta_Y=np.arctan2(np.sin(theta_Y_cm),(np.cos(theta_Y_cm)+gamma_))
            if theta_Y < 0:
                theta_Y += np.pi
    
            theta_b_cm_deg=theta_b_cm*180/np.pi
            theta_b_deg=theta_b*180/np.pi
            phi_b=phi_b_cm
            phi_Y=phi_Y_cm
            Tb_1=((np.sqrt(ma*mb*Ta)*np.cos(theta_b)+np.sqrt(ma*mb*Ta*(np.cos(theta_b))**2+(mY+mb)*(mY*Q+mY*Ta-ma*Ta)))/(mY+mb))**2
            Tb_2=((np.sqrt(ma*mb*Ta)*np.cos(theta_b)-np.sqrt(ma*mb*Ta*(np.cos(theta_b))**2+(mY+mb)*(mY*Q+mY*Ta-ma*Ta)))/(mY+mb))**2
            pb_1 = np.sqrt(2 * mb * Tb_1)
            pb_2 = np.sqrt(2 * mb * Tb_2)
    
            pb_1_z=pb_1*np.cos(theta_b)
            pb_1_x=pb_1*np.sin(theta_b)*np.cos(phi_b)
            pb_1_y=pb_1*np.sin(theta_b)*np.sin(phi_b)

            pb_2_z=pb_2*np.cos(theta_b)
            pb_2_x=pb_2*np.sin(theta_b)*np.cos(phi_b)
            pb_2_y=pb_2*np.sin(theta_b)*np.sin(phi_b)

            pY_1_z=pa-pb_1_z
            pY_1_x=-pb_1_x
            pY_1_y=-pb_1_y

            pY_2_z=pa-pb_2_z
            pY_2_x=-pb_2_x
            pY_2_y=-pb_2_y

            theta_Y1=np.arctan2(np.sqrt(pY_1_x*pY_1_x+pY_1_y*pY_1_y),pY_1_z)
            if theta_Y1 < 0:
                theta_Y1 += np.pi
                
            theta_Y2=np.arctan2(np.sqrt(pY_2_x*pY_2_x+pY_2_y*pY_2_y),pY_2_z)
            if theta_Y2 < 0:
                theta_Y2 += np.pi
            
            pY_1=np.sqrt(pY_1_z*pY_1_z+pY_1_x*pY_1_x+pY_1_y*pY_1_y)
            pY_2=np.sqrt(pY_2_z*pY_2_z+pY_2_x*pY_2_x+pY_2_y*pY_2_y)

            TY_1=(pY_1*pY_1)/(2*mY)
            TY_2=(pY_2*pY_2)/(2*mY)
            
            

            theta_Y_cm_deg=theta_Y_cm*180/np.pi
            theta_Y_deg=theta_Y*180/np.pi

            if abs(theta_Y-theta_Y1) < abs(theta_Y-theta_Y2):
                Tb = Tb_1
                TY = TY_1
            else:
                Tb = Tb_2
                TY = TY_2
            vb=np.sqrt(2*(Tb)/mb)
            vb_z=vb*np.cos(theta_b)
            vb_x=vb*np.sin(theta_b)*np.cos(phi_b)
            vb_y=vb*np.sin(theta_b)*np.sin(phi_b)
            Tc = Td = E_rel = theta_c = theta_d = None

            if breakup:
                vd_rest=np.sqrt((2*Q_break)/(md*(1+md/mc)))
                vc_rest=vd_rest*(md/mc)
                theta_d_rest=np.random.uniform(0,np.pi)
                phi_d_rest=np.random.uniform(0,2*np.pi)

                theta_c_rest=np.pi-theta_d_rest
                phi_c_rest=phi_d_rest+np.pi
                if phi_c_rest > 2*np.pi:
                    phi_c_rest -= 2*np.pi


                vd_rest_z=vd_rest*np.cos(theta_d_rest)
                vd_rest_x=vd_rest*np.sin(theta_d_rest)*np.cos(phi_d_rest)
                vd_rest_y=vd_rest*np.sin(theta_d_rest)*np.sin(phi_d_rest)

                vc_rest_z=vc_rest*np.cos(theta_c_rest)
                vc_rest_x=vc_rest*np.sin(theta_c_rest)*np.cos(phi_c_rest)
                vc_rest_y=vc_rest*np.sin(theta_c_rest)*np.sin(phi_c_rest)

                vd_z=vb_z+vd_rest_z
                vd_y=vb_y+vd_rest_y
                vd_x=vb_x+vd_rest_x

                vc_z=vb_z+vc_rest_z
                vc_y=vb_y+vc_rest_y
                vc_x=vb_x+vc_rest_x

                vc=np.sqrt(vc_x*vc_x+vc_y*vc_y+vc_z*vc_z)
                vd=np.sqrt(vd_x*vd_x+vd_y*vd_y+vd_z*vd_z)

                phi_c=np.arctan2(vc_y,vc_x)
                if phi_c < 0:
                    phi_c += 2*np.pi
                theta_c=np.arctan2(np.sqrt(vc_x*vc_x+vc_y*vc_y),vc_z)
                if theta_c < 0:
                    theta_c += np.pi

                phi_d=np.arctan2(vd_y,vd_x)
                if phi_d < 0:
                    phi_d += 2*np.pi
                theta_d=np.arctan2(np.sqrt(vd_x*vd_x+vd_y*vd_y),vd_z)
                if theta_d < 0:
                    theta_d += np.pi
                Tc=0.5*mc*vc*vc
                Td=0.5*md*vd*vd

                theta_rel = np.arccos(
                    np.cos(theta_c) * np.cos(theta_d) +
                    np.sin(theta_c) * np.sin(theta_d) * np.cos(phi_c - phi_d)
                )
                E_rel = (md*Tc + mc*Td - 2*np.sqrt(mc*Tc*md*Td)*np.cos(theta_rel)) / (mc + md)

            Tb_arr.append(Tb )
            TY_arr.append(TY )
            theta_b_deg_arr.append(theta_b_deg)
            theta_Y_deg_arr.append(theta_Y_deg)

            rows.append([
                theta_b_deg, Tb, theta_Y_deg, TY,
                np.degrees(theta_c) if breakup else '',
                Tc if breakup else '',
                np.degrees(theta_d) if breakup else '',
                Td if breakup else '',
                E_rel if breakup else ''
            ])

        # Plot
        plt.figure()
        plt.plot(np.array(theta_b_deg_arr),np.array(Tb_arr)/b.A,'o',markersize=3,label='b')
        plt.plot(np.array(theta_Y_deg_arr),np.array(TY_arr)/Y.A,'o',markersize=3,label='Y')
        plt.ylabel("Energy[Lab-MeV/u]")
        plt.xlabel("Angle[Lab-deg]")
        plt.legend()
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerows(rows)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode('utf-8')

        return jsonify({"image": img_base64, "csv": csv_base64,"Q": Q, "Ecm": Ecm, "vcb": vcb,})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/convert-angle', methods=['POST'])
def convert_angle():
    try:
        data = request.json
        a = Nucleus(data['a'])
        X = Nucleus(data['X'])
        b = Nucleus(data['b'])
        Y = Nucleus(data['Y'])
        Ta = float(data['Ta'])
        Ex_b = float(data['Ex_b'])
        Ex_Y = float(data['Ex_Y'])

        ma = a.mass 
        mX = X.mass 
        mb = b.mass 
        mY = Y.mass 

        Q = ((ma + mX) - (mb + mY)) - Ex_b - Ex_Y
        gamma = np.sqrt(((ma * mb) / (mX * mY)) * (Ta / (Ta + Q * (1 + ma / mX))))
        gamma_ = np.sqrt(((ma * mY) / (mX * mb)) * (Ta / (Ta + Q * (1 + ma / mX))))

        angle_type = data['angleType']
        angle_value = np.radians(float(data['angleValue']))

        if angle_type == 'theta_lab_b':
            theta_lab_b = angle_value
            theta_cm_b = np.arctan2(np.sin(theta_lab_b), np.cos(theta_lab_b) - gamma)
            theta_cm_Y = np.pi - theta_cm_b
            theta_lab_Y = np.arctan2(np.sin(theta_cm_Y), np.cos(theta_cm_Y) + gamma_)

        elif angle_type == 'theta_lab_Y':
            theta_lab_Y = angle_value
            theta_cm_Y = np.arctan2(np.sin(theta_lab_Y), np.cos(theta_lab_Y) - gamma_)
            theta_cm_b = np.pi - theta_cm_Y
            theta_lab_b = np.arctan2(np.sin(theta_cm_b), np.cos(theta_cm_b) + gamma)

        elif angle_type == 'theta_cm_b':
            theta_cm_b = angle_value
            theta_lab_b=np.arctan2(np.sin(theta_cm_b),(np.cos(theta_cm_b)+gamma))
            if theta_lab_b < 0:
                theta_lab_b += np.pi
            theta_cm_Y = np.pi - theta_cm_b
            theta_lab_Y=np.arctan2(np.sin(theta_cm_Y),(np.cos(theta_cm_Y)+gamma_))
            if theta_lab_Y < 0:
                theta_lab_Y += np.pi
            

        elif angle_type == 'theta_cm_Y':
            theta_cm_Y = angle_value 
            theta_lab_Y = np.arctan2(np.sin(theta_cm_Y),(np.cos(theta_cm_Y)+gamma_))
            if theta_lab_Y < 0:
                theta_lab_Y += np.pi
            theta_cm_b = np.pi - theta_cm_Y
            theta_lab_b = np.arctan2(np.sin(theta_cm_b),(np.cos(theta_cm_b)+gamma))
            if theta_lab_b < 0:
                theta_lab_b += np.pi
            

        return jsonify({
            "theta_lab_b": np.degrees(theta_lab_b),
            "theta_lab_Y": np.degrees(theta_lab_Y),
            "theta_cm_b": np.degrees(theta_cm_b),
            "theta_cm_Y": np.degrees(theta_cm_Y)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
