from flask import render_template, url_for, request, redirect
from webapp import app
from webapp.models import Patient, Bed
from webapp import db


# @app.route("/")
# def authentication_page():
#    return render_template('authentication_page.html')

@app.route("/", methods=['GET'])
def home():
    patients = Patient.query.all()
    return render_template('home.html', patients=patients)


@app.route('/<int:id>/delete', methods=['GET', 'POST'])
def delete_patient(id):
    patients = Patient.query.filter_by(patient_id=id).first()

    try:
        db.session.delete(patients)
        db.session.commit()
        return redirect('/')
    except:
        return "There was an error"


@app.route('/<int:id>/edit', methods=['GET', 'POST'])
def edit_patient(id):
    patient = Patient.query.filter_by(patient_id=id).first()

    if request.method == 'POST':
        if patient:
            patient.triage_category = request.form['triage_category']
            patient.status = request.form['status']
            if patient.patient_bed.ward is not request.form['ward']:
                patient.patient_bed.ward = request.form['ward']
                db.session.commit()
            if int(patient.patient_bed.bed_id) != int(request.form['bed_id']):
                print("Changing bed from: "+ str(patient.patient_bed.bed_id)+" to "+ str(request.form['bed_id']))
                patient.patient_bed = None
                new_bed = Bed(bed_id=request.form['bed_id'], ward=request.form['ward'])
                patient.patient_bed = new_bed
                db.session.commit()

        db.session.commit()
        return redirect('/')

    return render_template('edit.html', patient=patient)
