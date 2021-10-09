import React, { Component } from 'react';
import Select from 'react-select';

//gender
const genderValues = [
  { value: '1', label: 'Male' },
  { value: '0', label: 'Female' },
];

//graduated
const graduatedValues = [
  { value: '1', label: 'Yes' },
  { value: '0', label: 'No' },
];


const profession = [
  { value: 'Doctor', label: 'Doctor' },
  { value: 'Engineer', label: 'Engineer' },
  { value: 'Artist', label: 'Artist' },
  { value: 'Homemaker', label: 'Homemaker' },
  { value: 'Lawyer', label: 'Lawyer' },
  { value: 'Marketing', label: 'Marketing' },
  { value: 'Executive', label: 'Executive' },
  { value: 'Healthcare', label: 'Healthcare' },
  { value: 'Entertainment', label: 'Entertainment' },
];

const SpendingScore = [
  { value: 'Low', label: 'Low' },
  { value: 'Average', label: 'Average' },
  { value: 'High', label: 'High' },
];





export default class test extends Component {
  constructor(props) {
    super(props);
    this.onSelectGender = this.onSelectGender.bind(this);
    this.onSelectProfession = this.onSelectProfession.bind(this);
    this.onSelectGraduated = this.onSelectGraduated.bind(this);
    );
    this.state = {
      gender: '',
      profession: '',
      graduated:'',
    };
  }

  onSelectGender = (event) => {
    if (event) {
      console.log('value', parseInt(event.value));
      this.setState({ grade: event.value });
    }
  };

  onSelectProfession = (event) => {
    if (event) {
      if (event.value === 'Doctor') {
        const obj = {
          Doctor: 1,
          Engineer: 0,
          Artist: 0,
          Homemaker: 0,
          Lawyer: 0,
          Marketing: 0,
          Executive: 0,
          Healthcare: 0,
          Entertainment: 0,
         
        };
        console.log('Doc', obj);
      } else if (event.value === 'Engineer') {
        const obj = {
          Doctor: 0,
          Engineer: 1,
          Artist: 0,
          Homemaker: 0,
          Lawyer: 0,
          Marketing: 0,
          Executive: 0,
          Healthcare: 0,
          Entertainment: 0,
        };
        console.log('Eng', obj);

      } else if (event.value === 'Artist') {
        const obj = {
          Doctor: 0,
          Engineer: 0,
          Artist: 1,
          Homemaker: 0,
          Lawyer: 0,
          Marketing: 0,
          Executive: 0,
          Healthcare: 0,
          Entertainment: 0,
        };
        console.log('Artist', obj);
      }
       else if (event.value === 'Homemaker') {
        const obj = {
          Doctor: 0,
          Engineer: 0,
          Artist: 0,
          Homemaker: 1,
          Lawyer: 0,
          Marketing: 0,
          Executive: 0,
          Healthcare: 0,
          Entertainment: 0,
        };
        console.log('Homemaker', obj);
      }
      else if (event.value === 'Lawyer') {
        const obj = {
          Doctor: 0,
          Engineer: 0,
          Artist: 0,
          Homemaker: 0,
          Lawyer: 1,
          Marketing: 0,
          Executive: 0,
          Healthcare: 0,
          Entertainment: 0,
        };
        console.log('Law', obj);
      } 
       else if (event.value === 'Marketing') {
        const obj = {
          Doctor: 0,
          Engineer: 0,
          Artist: 0,
          Homemaker: 0,
          Lawyer: 0,
          Marketing: 1,
          Executive: 0,
          Healthcare: 0,
          Entertainment: 0,
        };
        console.log('Marketing', obj);
       }
        else if (event.value === 'Executive') {
        const obj = {
          Doctor: 0,
          Engineer: 0,
          Artist: 0,
          Homemaker: 0,
          Lawyer: 0,
          Marketing: 0,
          Executive: 1,
          Healthcare: 0,
          Entertainment: 0,
        };
        console.log('Executive', obj);
       }
       else if (event.value === 'Healthcare') {
        const obj = {
          Doctor: 0,
          Engineer: 0,
          Artist: 0,
          Homemaker: 0,
          Lawyer: 0,
          Marketing: 0,
          Executive: 0,
          Healthcare: 1,
          Entertainment: 0,
        };
        console.log('Healthcare', obj);
       }
       else if (event.value === 'Entertainment') {
        const obj = {
          Doctor: 0,
          Engineer: 0,
          Artist: 0,
          Homemaker: 0,
          Lawyer: 0,
          Marketing: 0,
          Executive: 0,
          Healthcare: 0,
          Entertainment: 1,
        };
        console.log('Entertainment', obj);
       }


  };

  onSelectGraduated = (event) => {
    if (event) {
      console.log('value', parseInt(event.value));
      this.setState({ grade: event.value });
    }
  };

  render() {
    return (
      <div>
        <h1>AutoMobile Company Customer Prediction</h1>
        <label>Gender</label>
        <Select
          options={genderValues}
          className="basic-single p-0 m-0"
          classNamePrefix="select"
          onChange={this.onSelectGender}
        />

        <label>Profession</label>
        <Select
          options={profession}
          className="basic-single p-0 m-0"
          classNamePrefix="select"
          onChange={this.onSelectProfession}
        />

        <label>Gender</label>
        <Select
          options={graduatedValues}
          className="basic-single p-0 m-0"
          classNamePrefix="select"
          onChange={this.onSelectGraduated}
        />


      </div>
    );
  }
}
