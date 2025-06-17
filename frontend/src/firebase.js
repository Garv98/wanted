// Firebase initialization for your React app

import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyBp5hvhwTLPGEol4HnHmP-8gk7A9FvIPOg",
  authDomain: "wanted-2faaa.firebaseapp.com",
  projectId: "wanted-2faaa",
  storageBucket: "wanted-2faaa.appspot.com",
  messagingSenderId: "823652845527",
  appId: "1:823652845527:web:b58cd080ae5e02f5f86d92",
  measurementId: "G-Z1MQHYVRBL"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

export { app, analytics };
