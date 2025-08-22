# TODO List - CAMD ISS Project

## Current Tasks

### High Priority
- [ ] Verify the antoine parameter calculator
- [ ] Simplify tge antoine parameter calculator

### Medium Priority
- [ ] 

### Low Priority
- [ ] make notebooks
- [ ] make automated tests

---

## Completed Tasks ✅
- [x] Verify the compound_validator.py
- [x] Simplify the compound_validator.py

### Property Calculator
- [x] Create PropertyCalculator class with Constantinou-Gani method
- [x] Add comprehensive test suite (test_property_calculator.py)
- [x] Create tutorial notebook with examples
- [x] Add units documentation to property methods
- [x] Verified that properties are calculted correctly

### Code Quality
- [x] Add missing groups handling and summary
- [x] Implement error handling for missing data
- [x] Add convenience functions

---

## Future Enhancements

### Property Calculator
- [ ] Add more properties (viscosity, surface tension, etc.)
- [ ] Implement other group contribution methods
- [ ] Performance optimization for large datasets

### UNIFAC Integration
- [ ] Connect PropertyCalculator with UNIFAC calculations
- [ ] Implement mixture property calculations
- [ ] Add phase equilibrium calculations

### CAMD Workflow
- [ ] Integrate with molecular design algorithms
- [ ] Add constraint handling
- [ ] Implement optimization routines

### Documentation
- [ ] Add API documentation
- [ ] Create user guide
- [ ] Add more tutorial examples

---

## Notes & Ideas

### Recent Observations
- PropertyCalculator works well with existing data structure
- Test coverage is comprehensive with specific expected values
- Notebook provides good examples for users

### Technical Debt
- Some commented code blocks in property_calculator.py should be cleaned up
- Consider adding input validation for compound dictionaries
- Error handling for negative/zero sums in logarithmic calculations

### Research Questions
- How to handle missing group contributions optimally?
- Which additional properties would be most valuable?
- Integration points with existing UNIFAC implementation?


---

*Last updated: [Current Date]*
*Project: Computer-Aided Molecular Design with Infinite Sharp Split (CAMD ISS)*